"""Offline overlapping-window deployment and controlled alignment replay.

Not incremental/causal neural inference: recompute a one-second window every
100 ms. Two successive hits trigger one event; 1 s cooldown. No real Pi timing
or continuous-environment false-activation guarantee is inferred from replay.
"""
import torch
import torch.nn.functional as F
from wakeword.data import LogMelFeatures, load_wav, fit_length
from wakeword.metrics import compute_metrics

STREAM_CONFIG = dict(window_ms=1000, stride_ms=100, consecutive_hits=2,
                     cooldown_ms=1000, replay_offsets_ms=[0, 250, 500, 750],
                     replay_duration_ms=2000, context="zero silence only",
                     score="max over adjacent pairs of min(sigmoid logits)",
                     metric_unit="recording-placement event; NOT clip FAR or real-world FAR/hour",
                     latency_note="recording end is not verified speech end")


class WindowFeatures(LogMelFeatures):
    def forward(self, waves):
        power = torch.stft(waves, self.n_fft, self.hop_length, self.win_length,
                           self.window, center=False, return_complex=True).abs().square()
        features = (self.mel_bank @ power).clamp_min(1e-10).log()
        return (features-features.mean(dim=(-2, -1), keepdim=True)) / features.std(
            dim=(-2, -1), keepdim=True).clamp_min(1e-5)


class StreamingDetector:
    """Arbitrary-size PCM chunks; return timestamped events every 100 ms.

    CPU tensors scaled to [-1,1], 16 kHz mono; persistent buffer + hysteresis.
    Silence prefill permits immediate scoring, without waiting a full second.
    """
    def __init__(self, model, threshold, device="cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.features = WindowFeatures().to(self.device)
        self.threshold = threshold
        self.buffer = torch.zeros(16000, device=self.device)
        self.pending = torch.empty(0, device=self.device)
        self.samples, self.hits, self.last_event = 0, 0, -16000

    @torch.inference_mode()
    def push(self, samples):
        self.pending = torch.cat((self.pending, samples.to(self.device).flatten()))
        events = []
        while self.pending.numel() >= 1600:
            chunk, self.pending = self.pending[:1600], self.pending[1600:]
            self.buffer = torch.cat((self.buffer[1600:], chunk))
            self.samples += 1600
            probability = float(self.model(self.features(self.buffer).unsqueeze(0).unsqueeze(0)).sigmoid()[0])
            self.hits = self.hits+1 if probability >= self.threshold else 0
            if self.hits >= 2 and self.samples-self.last_event >= 16000:
                self.last_event = self.samples
                events.append(dict(timestamp_seconds=self.samples/16000, probability=probability))
        return events


@torch.inference_mode()
def replay_scores(model, items, device):
    """Every example at four frozen offsets; raw placement/window scores.

    No waveform augmentation; evaluation does not mix training background noise.
    Test must only call this AFTER best reload and validation thresholds freeze.
    """
    model.eval()
    extractor = WindowFeatures().to(device)
    outputs, labels, identities = [], [], []
    for path, label in items:
        recording = fit_length(load_wav(path), False)
        for offset in [0, 4000, 8000, 12000]:
            stream = torch.zeros(32000)
            stream[offset:offset+16000] = recording
            windows = F.pad(stream, (16000, 0)).unfold(0, 16000, 1600)[1:].contiguous()
            probabilities = model(extractor(windows.to(device)).unsqueeze(1)).sigmoid().cpu()
            outputs.append(probabilities)
            labels.append(label)
            identities.append(dict(sample=path.parent.name+'/'+path.name, word=path.parent.name,
                                   offset_ms=offset/16, label=label))
    probabilities = torch.stack(outputs)
    # Debounce-aware event confidence, exactly two consecutive threshold hits.
    scores = torch.minimum(probabilities[:, :-1], probabilities[:, 1:]).amax(1)
    return scores, torch.tensor(labels), identities, probabilities


def replay_report(scores, labels, identities, probabilities, threshold):
    result = compute_metrics(scores, labels, threshold).to_dict()
    result['protocol'] = STREAM_CONFIG
    result['by_offset'] = {}
    for offset in [0, 250, 500, 750]:
        mask = torch.tensor([r['offset_ms'] == offset for r in identities])
        result['by_offset'][str(offset)] = compute_metrics(scores[mask], labels[mask], threshold).to_dict()
    detected = (scores >= threshold).reshape(-1, 4)
    positive_recordings = labels.reshape(-1, 4)[:, 0].bool()
    result['positive_recordings_detected_at_any_offset'] = int(detected[positive_recordings].any(1).sum())
    result['positive_recordings_detected_at_all_offsets'] = int(detected[positive_recordings].all(1).sum())
    result['positive_recordings'] = int(positive_recordings.sum())
    latencies = []
    for i, row in enumerate(identities):
        if not row['label']:
            continue
        hits = (probabilities[i, :-1] >= threshold) & (probabilities[i, 1:] >= threshold)
        indices = hits.nonzero().flatten()
        if len(indices):
            # Signed timing relative to one-second recording boundary, NOT true end of speech.
            latencies.append((int(indices[0])+2)*100 - (row['offset_ms']+1000))
    result['first_event_minus_recording_end_ms'] = latencies
    return result
