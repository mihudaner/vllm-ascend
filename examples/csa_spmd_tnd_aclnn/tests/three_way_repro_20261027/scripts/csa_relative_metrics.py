"""Report explicit element-count, absolute, and relative error metrics."""
import json
from pathlib import Path
import torch

root = Path('/data00/wangkai/aclnnWorkspace/compare-20261027')
results = {}
for case in json.loads((root / 'cases.json').read_text())['cases'][:8]:
    folder = root / (case['name'] + '_v2')
    actual = torch.load(folder / 'candidate.pt', map_location='cpu', weights_only=True)['output'].double()
    references = {
        'python_pto': torch.load(folder / 'reference_state.pt', map_location='cpu', weights_only=True)['output'][:case['valid_tokens']],
        'native': torch.load(folder / 'native_output.pt', map_location='cpu', weights_only=True),
        'golden': torch.load(folder / 'golden_output.pt', map_location='cpu', weights_only=True)['output'],
    }
    results[case['name']] = {}
    for name, reference in references.items():
        reference = reference.double()
        assert actual.shape == reference.shape
        assert torch.isfinite(actual).all() and torch.isfinite(reference).all()
        delta = (actual-reference).abs()
        zero = reference == 0
        zero_mismatch = int((zero & (delta != 0)).sum())
        relative = delta[~zero] / reference[~zero].abs()
        item = dict(elements=actual.numel(),
                    exceeded=int((delta > .01 + .01*reference.abs()).sum()),
                    exceeded_percent=float((delta > .01 + .01*reference.abs()).double().mean()*100),
                    max_absolute=float(delta.max()),
                    max_relative_percent_nonzero=float(relative.max()*100) if relative.numel() else 0,
                    reference_zero_count=int(zero.sum()), zero_reference_nonzero_actual=zero_mismatch)
        if relative.numel():
            index = int(relative.argmax())
            item['max_relative_pair'] = dict(actual=float(actual[~zero][index]),
                                            reference=float(reference[~zero][index]))
        results[case['name']][name] = item
(root / 'relative_metrics.json').write_text(json.dumps(results, indent=2) + '\n')
for case, values in results.items():
    print(case, {name: v['max_relative_pair'] for name, v in values.items() if name != 'python_pto'})
