import argparse
import os
import re
import pandas as pd


def sanitize_filename(name: str) -> str:
	"""Sanitize a filename base by replacing spaces and colons with underscores
	and removing any characters that are unsafe for filenames.
	"""
	# replace spaces and colons with underscore
	s = re.sub(r"[\s:]+", "_", name)
	# remove any characters except alphanumerics, underscore, hyphen
	s = re.sub(r"[^A-Za-z0-9_\-]", "", s)
	return s


def fill_csv(input_path: str, output_dir: str = ".") -> str:
	"""Read CSV at input_path, forward-fill missing values and save to output_dir.

	Returns the path to the saved filled CSV.
	"""
	if not os.path.exists(input_path):
		raise FileNotFoundError(f"Input file not found: {input_path}")

	# Read CSV
	df = pd.read_csv(input_path)

	# Forward-fill all columns so each row contains the last known values
	df_filled = df.ffill()

	# Build output filename based on input filename
	base = os.path.splitext(os.path.basename(input_path))[0]
	safe_base = sanitize_filename(base)
	out_name = f"{safe_base}_filled.csv"
	out_path = os.path.join(output_dir, out_name)

	# Save to output path
	df_filled.to_csv(out_path, index=False)
	return out_path


def main():
	parser = argparse.ArgumentParser(description="Forward-fill a CSV and save a filled copy.")
	parser.add_argument("file", nargs="?", default=None, help="Path to the CSV file to fill")
	parser.add_argument("--outdir", "-o", default='.', help="Directory to save the filled CSV (default: current directory)")

	args = parser.parse_args()

	file_name = "data.csv"

	try:
		saved = fill_csv(file_name, output_dir=args.outdir)
		print(f"Done! Saved filled CSV to: {saved}")
	except Exception as e:
		print(f"Error: {e}")


if __name__ == "__main__":
	main()