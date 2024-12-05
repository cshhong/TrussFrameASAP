"""
For result images in a Dropbox folder,
scrape the direct image links (ending with raw=1 as opposed to dl=0)
Return a csv file with columns [idx, img_url] that are ordered by idx
"""

import dropbox
import csv
import re
import copy
import pandas as pd
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

def convert_to_raw_link(url):
    """
    Converts a Dropbox shared link with `dl=0` to a direct raw link with `raw=1`.

    Parameters:
        url (str): The original Dropbox URL.

    Returns:
        str: The updated URL with `raw=1`.
    """
    # Parse the URL
    parsed_url = urlparse(url)

    # Parse the query parameters
    query_params = parse_qs(parsed_url.query)

    # Remove 'dl=0' and add 'raw=1'
    query_params.pop('dl', None)
    query_params['raw'] = '1'

    # Rebuild the query string
    updated_query = urlencode(query_params, doseq=True)

    # Reconstruct the URL
    updated_url = urlunparse((
        parsed_url.scheme,       # Scheme (e.g., https)
        parsed_url.netloc,       # Network location (e.g., www.dropbox.com)
        parsed_url.path,         # Path (e.g., /scl/fi/...)
        parsed_url.params,       # Parameters (not used)
        updated_query,           # Updated query string
        parsed_url.fragment      # Fragment (not used)
    ))

    return updated_url

def get_or_create_shared_link(dbx, file_path):
    """
    Get an existing shared link for a file or create a new one if it doesn't exist.

    Parameters:
        dbx (dropbox.Dropbox): Dropbox client object.
        file_path (str): Path to the file in Dropbox.

    Returns:
        str: A Dropbox shared link (converted to raw=1 for direct use).
    """
    try:
        # Check if a shared link already exists for the file
        shared_links = dbx.sharing_list_shared_links(path=file_path, direct_only=True).links
        if shared_links:
            # Use the existing shared link
            shared_link = shared_links[0].url
        else:
            # Create a new shared link
            shared_link = dbx.sharing_create_shared_link_with_settings(file_path).url

        # Convert the shared link to use `raw=1` for direct access
        raw_shared_link = convert_to_raw_link(shared_link)
        return raw_shared_link

    except dropbox.exceptions.ApiError as e:
        print(f"Error handling file {file_path}: {e}")
        return None

def scrape_image_links(access_token, dropbox_folder, output_csv):
    """
    Scrapes direct image links from a Dropbox folder and saves to a CSV.

    Parameters:
        access_token (str): Dropbox API access token.
        dropbox_folder (str): Path to the Dropbox folder.
        output_csv (str): Output CSV file name (must contain 'idx' and 'img_url' columns, idx values are prepopulated from hdf5 file.

    Returns:
        None
    """

    # Initialize Dropbox client
    dbx = dropbox.Dropbox(access_token)

    # Read the existing CSV into a DataFrame
    try:
        df = pd.read_csv(output_csv)
    except FileNotFoundError:
        print(f"Error: File '{output_csv}' not found.")
        return

    # Ensure required columns exist
    if "idx" not in df.columns or "img_url" not in df.columns:
        print("Error: CSV file must contain 'idx' and 'img_url' columns.")
        return


    # List folder contents
    try:
        result = dbx.files_list_folder(dropbox_folder)
    except dropbox.exceptions.ApiError as e:
        print(f"Error accessing Dropbox folder: {e}")
        return

    links_dict = {}
    for entry in result.entries:
        if isinstance(entry, dropbox.files.FileMetadata):
            file_path = entry.path_lower
            # Get or create the shared link
            shared_link = get_or_create_shared_link(dbx, file_path)
            if shared_link:
                # Extract the index from the filename
                match = re.match(r"end_(\d+)\.\w+", entry.name)
                if match:
                    idx = int(match.group(1))  # Convert to int for consistent indexing
                    links_dict[idx] = shared_link

    # Populate the 'img_url' column in the DataFrame
    df["img_url"] = df["idx"].map(links_dict).fillna(df["img_url"])  # Keep existing links if not found

    # Save the updated DataFrame back to the CSV
    df.to_csv(output_csv, index=False)
    print(f"Updated image links saved to {output_csv}")

# test the script
if __name__ == "__main__":
    # # Example URL
    # url = "https://www.dropbox.com/scl/fi/2aaaq6obe14i97td65va2/end_10.png?rlkey=5gn19n0iq1uxetkm0o6ldgrqy&dl=0"

    # # Convert to raw link
    # raw_url = convert_to_raw_link(url)
    # print(raw_url)

    # Dropbox API Access Token
    # API Console : https://www.dropbox.com/developers/apps/info/a44uvsxdkczevsi
    ACCESS_TOKEN = "sl.CBHYw6fev6bHOGvo4GrTJIvBhsQcGCsVCidh_FzThS_l2uh6NYfvIYDLuH8N8SFuhzYPSgq8bx2YtYg5KgntGxvh4QwauYF74HU9pzoJ7HuS1PrPw7_LwWcjRgnX54Oilw4i1irk_bgdTf8"

    # Dropbox Folder Path
    DROPBOX_FOLDER = "/test_umap" # relative path within Dropbox/Apps/TrussFrame/

    # Output CSV File
    OUTPUT_CSV = "test_image_urls_real.csv"

    scrape_image_links(ACCESS_TOKEN, DROPBOX_FOLDER, OUTPUT_CSV)
    # sort the csv file by idx
    

    
    