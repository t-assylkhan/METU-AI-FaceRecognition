import os
import urllib.request
import bz2

def main():
    """
    Download and extract the shape predictor file if it doesn't exist
    """
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    
    # Check if file already exists
    if os.path.isfile(predictor_path):
        print(f"File {predictor_path} already exists. Skipping download.")
        return
    
    # URL of the compressed file
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    compressed_path = predictor_path + ".bz2"
    
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, compressed_path)
    
    print("Extracting file...")
    with open(predictor_path, 'wb') as new_file, bz2.BZ2File(compressed_path, 'rb') as file:
        for data in iter(lambda: file.read(100 * 1024), b''):
            new_file.write(data)
    
    # Remove the compressed file
    os.remove(compressed_path)
    print("Download and extraction complete!")

if __name__ == "__main__":
    main()