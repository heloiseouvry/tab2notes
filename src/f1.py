import argparse
import preprocess

parser = argparse.ArgumentParser()
parser.add_argument('-i','--input', help='path to input image file (required)')
parser.add_argument('-d','--dest', help='path to output image file')
parser.add_argument('-v','--verbose', help='more verbose')
args = parser.parse_args()

if __name__ == "__main__":
    img = preprocess.read_image(args.input, 0)