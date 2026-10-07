"""Load a trained checkpoint and classify a new response."""

import argparse
import json

import torch

from .data import input_tokens
from .model import TextCNN


def predict(checkpoint_path, response, prompt=None):
    if not isinstance(response, str) or not response.strip():
        raise ValueError('Provide a non-empty response.')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    config = checkpoint['preprocessing']
    if config['mode'] == 'prompt_response' and (not isinstance(prompt, str) or not prompt.strip()):
        raise ValueError('This checkpoint requires the original prompt as well as the response.')
    tokens = input_tokens({'response': response, 'prompt': prompt or ''}, **config)
    ids = torch.tensor([[checkpoint['vocab'].get(token, 1) for token in tokens] or [1]])
    model = TextCNN(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    with torch.inference_mode():
        probabilities = model(ids, torch.tensor([ids.size(1)])).softmax(1)[0].tolist()
    scores = dict(zip(checkpoint['labels'], probabilities))
    return {'family': max(scores, key=scores.get), 'probabilities': scores}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--response', required=True)
    parser.add_argument('--prompt')
    args = parser.parse_args(argv)
    print(json.dumps(predict(args.checkpoint, args.response, args.prompt), indent=2))


if __name__ == '__main__':
    main()
