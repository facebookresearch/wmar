# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Chameleon License found in the
# LICENSE file in the root directory of this source tree.

import base64
import io
import json
import math
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Union

import PIL
import torch
from PIL.Image import Image
from tokenizers import Tokenizer
from transformers import (
    LogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
    enable_full_determinism,
)

from deps.chameleon.inference import loader
from deps.chameleon.inference.alignment import AlignPromptRight
from deps.chameleon.inference.generation import ChameleonGenerator
from deps.chameleon.inference.image_tokenizer import ImageTokenizer
from deps.chameleon.inference.logits_processor import (
    AllowOnlyTokensLogitsProcessor,
    DisallowTokensAtOrAfterIndexLogitsProcessor,
    InBatchInstructCFGLogitsProcessor,
)
from deps.chameleon.inference.model_adapter import ChameleonModelAdapter
from deps.chameleon.inference.stopping_criteria import (
    MaxLengthCriteria,
    StopOnEOSAfterBatchIndex,
)
from deps.chameleon.inference.token_selector import (
    ArgmaxTokenSelector,
    MultinomialTokenSelector,
    ReplicatedInputTokenSelector,
)
from deps.chameleon.inference.transformer import Transformer
from deps.chameleon.inference.utils import DynamicGenerator, advance
from deps.chameleon.inference.vocab import VocabInfo, VocabTranslation


@dataclass
class Options:
    @dataclass
    class Text:
        repetition_penalty: float = 1.2
        temp: float = 0.7
        top_p: float = 0.9
        greedy: bool = False

    @dataclass
    class Image:
        @dataclass
        class CFG:
            guidance_scale_text: float = 3.0
            guidance_scale_image: float = 1.2

        cfg: CFG = field(default_factory=CFG)
        temp: float = 0.7
        top_p: float = 0.9
        greedy: bool = False

    max_seq_len: int = 4096
    max_gen_len: int = 4096
    seed: int | None = None
    txt: Text | bool = True
    img: Image | bool = True
    # extra_eos_tokens: list[int | str] = field(default_factory=lambda: ["<racm3:break>"])
    extra_eos_tokens: list[int | str] = field(default_factory=lambda: [])

    def __post_init__(self):
        if self.txt is True:
            self.txt = Options.Text()
        if self.img is True:
            self.img = Options.Image()


class TokenManager:
    def __init__(
        self,
        tokenizer_path: str,
        vqgan_cfg_path: str,
        vqgan_ckpt_path: str,
        device: str | None = None,
    ):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.vocab = VocabInfo(json.load(open(tokenizer_path))["model"]["vocab"])
        self.translation = VocabTranslation(self.vocab, device=device)
        self.image_tokenizer = ImageTokenizer(
            cfg_path=vqgan_cfg_path, ckpt_path=vqgan_ckpt_path, device=device
        )

    def pil_from_bpe_tokens(self, bpe_tokens: torch.Tensor) -> PIL.Image:
        image_tensor = self.translation.convert_bpe2img(bpe_tokens)
        if image_tensor.shape[0] < 1024:
            padding = (
                torch.ones(
                    [1024 - image_tensor.shape[0]],
                    dtype=int,
                    device=image_tensor.device,
                )
                * image_tensor[0]
            )
            image_tensor = torch.cat((image_tensor, padding)).unsqueeze(0)

        return self.image_tokenizer.pil_from_img_toks(image_tensor)

    def png_from_bpe_tokens(self, bpe_tokens: torch.Tensor) -> bytes:
        pil = self.pil_from_bpe_tokens(bpe_tokens)
        img_io = io.BytesIO()
        pil.save(img_io, format="PNG")
        return img_io.getvalue()

    def tokenize_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def tokenize_image(self, img: Image) -> list[int]:
        return (
            [self.vocab.begin_image]
            + self.translation.convert_img2bp2(
                self.image_tokenizer.img_tokens_from_pil(img)
            ).tolist()
            + [self.vocab.end_image]
        )

    def tokenize_b64img(self, b64img: str) -> list[int]:
        image_data = base64.b64decode(b64img)
        image_file = io.BytesIO(image_data)
        return self.tokenize_image(PIL.Image.open(image_file))

    def tokens_from_ui(self, inputs: list[dict]) -> list[int]:
        tokens = [self.vocab.bos_id]
        for input_ in inputs:
            if input_["type"] == "text":
                tokens += self.tokenize_text(input_["value"])
            elif input_["type"] == "image":
                if isinstance(input_["value"], str):
                    if input_["value"].startswith("data:"):
                        # Value Format: 'data:image/[^;]+;base64,[A-Za-z0-9+/]+={0,2}'
                        tokens += self.tokenize_b64img(input_["value"].split(",", 1)[1])
                    elif input_["value"].startswith("file:"):
                        tokens += self.tokenize_image(
                            PIL.Image.open(input_["value"].split(":", 1)[1])
                        )
                    else:
                        raise ValueError("Unknown image format.")
                elif isinstance(input_["value"], Image):
                    tokens += self.tokenize_image(input_["value"])
                else:
                    raise ValueError("Unknown image type.")
            elif input_["type"] == "sentinel":
                tokens += [
                    {
                        "<START-OF-IMAGE>": self.vocab.begin_image,
                        "<END-OF-TURN>": self.vocab.eot_id,
                    }[input_["value"]]
                ]
            elif input_["type"] == "ids":
                tokens += input_["value"]
            else:
                raise ValueError("Unknown input type.")
        return tokens

    def decode_text(self, ids: torch.LongTensor | list[list[int]]) -> list[str]:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        for row, values in enumerate(ids):
            try:
                ids[row] = values[: values.index(self.vocab.eos_id)]
            except ValueError:
                pass

        return self.tokenizer.decode_batch(ids)

    def decode_image(self, ids: torch.LongTensor) -> list[PIL.Image]:
        return [self.pil_from_bpe_tokens(sample) for sample in ids]


@dataclass
class DecodePiece:
    token: ChameleonGenerator.Token
    next_decoder: type["Decoder"] | None


class Decoder:
    def __init__(
        self,
        model: Transformer,
        vocab: VocabInfo,
        options: Options,
        input_ids: list[int],
    ): ...

    def __next__(self) -> DecodePiece: ...


class TextDecoder(Decoder):
    def __init__(
        self,
        model: Transformer,
        vocab: VocabInfo,
        options: Options,
        input_ids: list[list[int]],
    ):
        self.vocab = vocab
        self.options = options
        assert vocab.eos_id is not None

        prompt_lens = [len(inp) for inp in input_ids]
        max_prompt_len = max(prompt_lens)
        max_seq_len = min(options.max_seq_len, max_prompt_len + options.max_gen_len)

        self.eos_ids = [vocab.eos_id]
        for extra_eos_token in options.extra_eos_tokens:
            if isinstance(extra_eos_token, str):
                extra_eos_token = vocab.name2val[extra_eos_token]
            assert isinstance(extra_eos_token, int)
            self.eos_ids.append(extra_eos_token)

        stopping_criteria = [
            MaxLengthCriteria(max_seq_len),
        ] + [StopOnEOSAfterBatchIndex(eos_id, [max_prompt_len] * len(prompt_lens)) for eos_id in self.eos_ids]

        self.gen = ChameleonGenerator(
            model=ChameleonModelAdapter(model, max_seq_len=max_seq_len),
            input_ids=input_ids,
            stopping_criteria=stopping_criteria,
            logits_processors=self._logits_processors(),
            alignment=AlignPromptRight(vocab.pad_id),
            token_selector=(
                ArgmaxTokenSelector()
                if options.txt.greedy
                else MultinomialTokenSelector()
            ),
        )
        advance(self.gen, max_prompt_len)

    def _allowed_tokens(self) -> list[int]:
        allowed_tokens = [self.vocab.eos_id]
        if self.options.txt:
            allowed_tokens += self.vocab.text_tokens
        if self.options.img:
            allowed_tokens += [self.vocab.begin_image]
        return allowed_tokens

    def _logits_processors(self) -> list[LogitsProcessor]:
        logits_processors = [
            AllowOnlyTokensLogitsProcessor(self._allowed_tokens()),
        ]
        if isinstance(self.options.img, Options.Image):
            logits_processors += [
                DisallowTokensAtOrAfterIndexLogitsProcessor(
                    [self.vocab.begin_image],
                    self.options.max_seq_len - 1026,
                ),
            ]
        if isinstance(self.options.txt, Options.Text):
            logits_processors += [
                RepetitionPenaltyLogitsProcessor(self.options.txt.repetition_penalty),
                TemperatureLogitsWarper(self.options.txt.temp),
                TopPLogitsWarper(self.options.txt.top_p),
            ]
        return logits_processors

    def __next__(self) -> DecodePiece:
        tok = next(self.gen)
        next_decoder = None
        if (
            self.vocab.begin_image not in self.eos_ids
            and (tok.id == self.vocab.begin_image).all()
        ):
            next_decoder = ImageDecoder
        return DecodePiece(tok, next_decoder)


class ImageDecoder(Decoder):
    def __init__(
        self,
        model: Transformer,
        vocab: VocabInfo,
        options: Options,
        input_ids: list[list[int]],
    ):
        assert isinstance(options.img, Options.Image)
        self.vocab = vocab
        self.options = options
        self.batch_size = len(input_ids)
        logits_processors = [
            InBatchInstructCFGLogitsProcessor(
                options.img.cfg.guidance_scale_text,
                options.img.cfg.guidance_scale_image,
            ),
            AllowOnlyTokensLogitsProcessor(vocab.image_tokens),
            TemperatureLogitsWarper(options.img.temp),
            TopPLogitsWarper(options.img.top_p),
        ]

        for inp in input_ids:
            if inp[-1] != self.vocab.begin_image:
                inp.append(self.vocab.begin_image)

        max_prompt_len = max(len(inp) for inp in input_ids)
        self.gen = ChameleonGenerator(
            model=ChameleonModelAdapter(model, max_seq_len=max_prompt_len + 1024),
            input_ids=self._split_inputs_for_cfg(input_ids),
            logits_processors=logits_processors,
            alignment=AlignPromptRight(vocab.pad_id),
            token_selector=ReplicatedInputTokenSelector(
                (
                    ArgmaxTokenSelector()
                    if options.img.greedy
                    else MultinomialTokenSelector()
                ),
                n=3,
            ),
        )
        advance(self.gen, max_prompt_len)
        self.gen_count = 0

    def _split_inputs_for_cfg(self, input_ids: list[list[int]]) -> list[list[int]]:
        image_conditioned_allowed = set(self.vocab.image_tokens) | {
            self.vocab.bos_id,
            self.vocab.begin_image,
            self.vocab.end_image,
        }

        full_conditioned = input_ids

        image_conditioned = [
            [id for id in sample if id in image_conditioned_allowed]
            for sample in input_ids
        ]

        unconditioned = [
            [
                self.vocab.bos_id,
                self.vocab.begin_image,
            ]
        ] * self.batch_size

        return full_conditioned + image_conditioned + unconditioned

    def __next__(self) -> DecodePiece:
        if self.gen_count == 1024:
            id = torch.tensor([self.vocab.end_image] * self.batch_size)
            logits = torch.full(
                (self.batch_size, len(self.vocab.all_tokens)), -math.inf
            )
            logits[:, self.vocab.end_image] = 0
            return DecodePiece(
                ChameleonGenerator.Token(id=id, logits=logits),
                TextDecoder,
            )

        tok = next(self.gen)
        tok.id = tok.id.chunk(3)[0]
        self.gen_count += 1
        return DecodePiece(tok, None)


class Generator(Decoder):
    def __init__(
        self,
        model: Transformer,
        vocab: VocabInfo,
        options: Options,
        input_ids: list[list[int]],
    ):
        if options.seed is not None:
            enable_full_determinism(options.seed, warn_only=True)

        self.model = model
        self.vocab = vocab
        self.input_ids = input_ids[:]
        self.generated_token_ids: list[torch.LongTensor] = []
        self.options = options
        if not self.options.txt:
            self.dyngen = DynamicGenerator(
                ImageDecoder(model, vocab, options, input_ids)
            )
        else:
            self.dyngen = DynamicGenerator(
                TextDecoder(model, vocab, options, input_ids)
            )

    def __iter__(self):
        return self

    def __next__(self) -> ChameleonGenerator.Token:
        piece = next(self.dyngen)
        print(f"Got token {piece}")
        self.generated_token_ids.append(piece.token.id)
        if piece.next_decoder is not None:
            if not self.options.txt:
                raise StopIteration

            self.input_ids = [
                old_list + generated
                for old_list, generated in zip(
                    self.input_ids, torch.stack(self.generated_token_ids).T.tolist()
                )
            ]
            self.generated_token_ids = []
            self.dyngen.gen = piece.next_decoder(
                self.model,
                self.vocab,
                self.options,
                self.input_ids,
            )
        return piece.token


# Simplified single-GPU inference without distribution
class ChameleonInferenceModel:
    def __init__(
        self,
        model: str | object,
        tokenizer_path: str,
        vqgan_cfg_path: str,
        vqgan_ckpt_path: str | None = None,
        *,
        options: Options | None = None,
    ):
        self.options = options or Options()
        # setup tokenizer and image conversion
        self.token_manager = TokenManager(
            tokenizer_path, vqgan_cfg_path, vqgan_ckpt_path, device="cuda"
        )
        self.vocab = self.token_manager.vocab
        # load and move model to single GPU
        if isinstance(model, str):
            self.model = loader.load_model(model)
        else:
            self.model = model
        self.model.to("cuda")
        self.model.eval()

        # change default device to GPU
        torch.set_default_device("cuda")

    def generate(
        self,
        *,
        batch_prompt_ui: list[list[dict]],
        options: Options | None = None,
    ) -> torch.LongTensor:
        # batch_prompt_ui: list of lists of {type, value}
        opts = options or self.options
        # prepare token IDs
        batch_input_ids = [self.token_manager.tokens_from_ui(ui) for ui in batch_prompt_ui]
        # sequential generator
        gen = Generator(self.model, self.vocab, opts, batch_input_ids)
        toks = [piece.id for piece in gen]
        if not toks:
            return torch.LongTensor()
        return torch.stack(toks).T

    def decode_text(self, ids: torch.LongTensor | list[list[int]]) -> list[str]:
        return self.token_manager.decode_text(ids)

    def decode_image(self, ids: torch.LongTensor) -> list[torch.Tensor | object]:
        return self.token_manager.decode_image(ids)
