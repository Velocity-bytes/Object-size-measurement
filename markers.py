import cv2
import cv2.aruco as aruco
import numpy as np

def generate_aruco_marker(dictionary_name, marker_id, marker_size):
    """
    Generates an ArUco marker and saves it as a PNG image.

    Args:
        dictionary_name (str): The name of the ArUco dictionary to use
                               (e.g., "DICT_6X6_250").
        marker_id (int): The ID of the marker to generate.
        marker_size (int): The size of the marker in pixels (e.g., 200).
    """
    # Load the dictionary
    if dictionary_name == "DICT_4X4_50":
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    elif dictionary_name == "DICT_4X4_100":
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
    elif dictionary_name == "DICT_4X4_250":
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    elif dictionary_name == "DICT_4X4_1000":
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    elif dictionary_name == "DICT_5X5_50":
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
    elif dictionary_name == "DICT_5X5_100":
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
    elif dictionary_name == "DICT_5X5_250":
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
    elif dictionary_name == "DICT_5X5_1000":
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
    elif dictionary_name == "DICT_6X6_50":
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
    elif dictionary_name == "DICT_6X6_100":
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
    elif dictionary_name == "DICT_6X6_250":
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    elif dictionary_name == "DICT_6X6_1000":
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
    elif dictionary_name == "DICT_7X7_50":
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_7X7_50)
    elif dictionary_name == "DICT_7X7_100":
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_7X7_100)
    elif dictionary_name == "DICT_7X7_250":
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_7X7_250)
    elif dictionary_name == "DICT_7X7_1000":
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_7X7_1000)
    elif dictionary_name == "DICT_ARUCO_ORIGINAL":
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
    elif dictionary_name == "DICT_APRILTAG_16h5":
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_16h5)
    elif dictionary_name == "DICT_APRILTAG_25h9":
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_25h9)
    elif dictionary_name == "DICT_APRILTAG_36h10":
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h10)
    elif dictionary_name == "DICT_APRILTAG_36h11":
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    else:
        print(f"Error: Unknown dictionary name '{dictionary_name}'.")
        return

    # Create an image for the marker
    marker_image = np.zeros((marker_size, marker_size), dtype=np.uint8)
    marker_image = aruco.generateImageMarker(aruco_dict, marker_id, marker_size, marker_image, 1)

    # Save the marker image
    output_filename = f"aruco_marker_{marker_id}_{dictionary_name}.png"
    cv2.imwrite(output_filename, marker_image)
    print(f"Generated {output_filename}")

    # Display the marker
    cv2.imshow(f"ArUco Marker ID: {marker_id}", marker_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage:
    # You can change these values to generate different markers
    dictionary_type = "DICT_6X6_250"
    marker_identifier = 23
    marker_width = 200 # pixels

    generate_aruco_marker(dictionary_type, marker_identifier, marker_width)