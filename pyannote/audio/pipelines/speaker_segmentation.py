from typing import Callable, Optional, Text, Union

import torch
from pyannote.core import Annotation, SlidingWindow, SlidingWindowFeature

from pyannote.audio import Audio, Model, Pipeline
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines.utils import (
    PipelineModel,
    SpeakerDiarizationMixin,
    get_model,
)
from pyannote.audio.utils.permutation import permutate
from pyannote.audio.utils.powerset import Powerset


class StreamingSpeakerSegmentation(SpeakerDiarizationMixin, Pipeline):
    def __init__(
        self,
        segmentation: PipelineModel = "pyannote/segmentation",
        device: torch.device = None,
        use_auth_token: Union[Text, None] = None,
    ):
        super().__init__()

        self.segmentation = segmentation

        self.segmentation_model: Model = get_model(
            segmentation, use_auth_token=use_auth_token
        )
        specifications = self.segmentation_model.specifications
        if not specifications.powerset:
            raise ValueError(
                "Streaming speaker segmentation is only supported for powerset models."
            )
        if not specifications.duration == 10.0:
            raise ValueError(
                "Streaming speaker segmentation is only supported for models with duration=10."
            )
        self.latency = 2.5

        if device is None:
            device = self.segmentation_model.device
        self.device = device

        self._powerset = Powerset(
            len(specifications.classes), specifications.powerset_max_classes
        )
        self._powerset.to(self.device)

        self._frames: SlidingWindow = self.segmentation_model.introspection.frames
        self._audio = Audio(
            sample_rate=self.segmentation_model.hparams.sample_rate, mono="downmix"
        )

    def to(self, device: torch.device):
        """Send internal model to `device`"""

        self.segmentation_model.to(device)
        self._powerset.to(device)
        self.device = device
        return self

    def estimate_shift(self, current_start: float):
        pass

    def apply_permutation(self, prediction, permutation):
        pass

    def apply(
        self,
        file: AudioFile,
        hook: Optional[Callable] = None,
    ) -> Annotation:
        duration = self.segmentation_model.specifications.duration
        chunks = SlidingWindow(start=0, duration=duration, step=self.latency)

        previous_multilabel_hard_prediction = None
        guide = None
        for c, chunk in enumerate(chunks):
            waveform: torch.Tensor = self._audio.crop(file, chunk, duration=duration)[0]
            waveform = waveform.to(self.device)

            # apply model on current chunk and convert to hard multilabel prediction
            powerset_soft_prediction = self.segmentation_model(
                waveform[None], guide=guide
            )
            powerset_hard_prediction = torch.nn.functional.one_hot(
                torch.argmax(powerset_soft_prediction, dim=-1),
                self._powerset.num_powerset_classes,
            ).float()
            multilabel_hard_prediction = self._powerset.to_multilabel(
                powerset_hard_prediction
            )

            # permutate current prediction to match previous one
            if c > 0:
                # estimate number of frames between current and previous chunk
                latency = self.estimate_shift(chunk.start)

                # permutate current prediction to match previous one
                _previous = previous_multilabel_hard_prediction[:, latency:]
                _current = multilabel_hard_prediction[:, :-latency]
                _, permutation = permutate(_previous, _current)

                # check whether a new speaker has been detected
                # and update permutation to actually add one dimension if that is the case
                permutation = permutation  # TODO

                # apply permutation to current prediction
                multilabel_hard_prediction = self.apply_permutation(
                    multilabel_hard_prediction, permutation
                )

                # grab the frames that needs to be output
                yield SlidingWindowFeature(multilabel_hard_prediction[:, :-latency])

            else:
                yield

            # prepare guide for next chunk
            guide = torch.zeros_like(multilabel_hard_prediction)
            num_frames_per_chunk = multilabel_hard_prediction.shape[1]
            half = num_frames_per_chunk // 2
            guide[:, :half] = 2 * (
                multilabel_hard_prediction[
                    :, num_frames_per_chunk // 4 : num_frames_per_chunk // 4 + half
                ]
                - 0.5
            )
