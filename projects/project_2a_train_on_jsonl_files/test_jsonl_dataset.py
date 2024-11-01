import unittest
import argparse
from jsonl_dataset import JSONLDataset

class TestJSONLDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set this path via command-line argument
        parser = argparse.ArgumentParser(description="Path to JSONL file for testing JSONLDataset")
        parser.add_argument('--path_to_jsonl', type=str, required=True, help="Path to the JSONL file")
        args, _ = parser.parse_known_args()
        cls.path_to_jsonl = args.path_to_jsonl

    @unittest.skip("Runs in about 40 seconds")
    def test_length(self):
        """
        Test that the length of the dataset is correct.
        """
        dataset = JSONLDataset(self.path_to_jsonl)
        self.assertEqual(len(dataset), 7023114)

    def test_item_0(self):
        """
        Test that the first item in the dataset is correct.
        """
        dataset = JSONLDataset(self.path_to_jsonl)
        self.assertEqual(dataset[0], "Dating in New York constantly proves to be a real adventure. The spectacle of dating in the nightlife scene develops off of the mystery of it all --- the darkness, the glamor, the unpredictability. Like any game, dating a New Yorker has its rules, the key one being never to roll up to a club with any expectations. Keep reading to see our top five tips for dating a New Yorker.")

    def test_item_1(self):
        """
        Test that the second item in the dataset is correct.
        """
        dataset = JSONLDataset(self.path_to_jsonl)
        self.assertEqual(dataset[1], "function net = get_net(varargin)\n% net = get_net(model_file, phase_name) or\n% net = get_net(model_file, weights_file, phase_name)\n%   Construct a net from model_file, and load weights from weights_file\n%   phase_name can only be 'train' or 'test'\n\nCHECK(nargin == 2 || nargin == 3, ['usage: ' ...\n  'net = get_net(model_file, phase_name) or ' ...\n  'net = get_net(model_file, weights_file, phase_name)']);\nif nargin == 3\n  model_file = varargin{1};\n  weights_file = varargin{2};\n  phase_name = varargin{3};\nelseif nargin == 2\n  model_file = varargin{1};\n  phase_name = varargin{2};\nend\n\nCHECK(ischar(model_file), 'model_file must be a string');\nCHECK(ischar(phase_name), 'phase_name must be a string');\nCHECK_FILE_EXIST(model_file);\nCHECK(strcmp(phase_name, 'train') || strcmp(phase_name, 'test'), ...\n  sprintf('phase_name can only be %strain%s or %stest%s', ...\n  char(39), char(39), char(39), char(39)));\n\n% construct caffe net from model_file\nhNet = caffe_('get_net', model_file, phase_name);\nnet = caffe.Net(hNet);\n\n% load weights from weights_file\nif nargin == 3\n  CHECK(ischar(weights_file), 'weights_file must be a string');\n  CHECK_FILE_EXIST(weights_file);\n  net.copy_from(weights_file);\nend\n\nend\n")

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)  # Prevents unittest from interpreting command-line args
