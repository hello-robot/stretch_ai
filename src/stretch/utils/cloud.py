# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import io
import os
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaInMemoryUpload, MediaIoBaseDownload
from PIL import Image

from stretch.utils.memory import get_path_to_default_credentials

default_credentials_path = get_path_to_default_credentials()
default_folder_name = "hello_robot_uploads"


class GoogleDriveUploader:
    """
    A class to handle uploading photos to Google Drive.
    """

    def __init__(self, credentials_path: Optional[str] = None, folder_name: Optional[str] = None):
        """
        Initialize the GoogleDriveUploader.

        Args:
            credentials_path (str): Path to the service account JSON file.
            folder_id (str): ID of the Google Drive folder to upload to.
        """
        if credentials_path is None:
            credentials_path = default_credentials_path
        self.credentials_path = credentials_path
        if folder_name is None:
            folder_name = default_folder_name

        # Create a Google Drive service object
        self.drive_service = self._create_drive_service()

        # Look for the folder in Google Drive
        # This is where the photos will be uploaded
        self.folder_name = folder_name
        if not self.folder_exists(folder_name):
            print(f"Folder '{folder_name}' not found. Creating a new folder.")
            self.create_folder(folder_name)
        info = self.get_folder_by_name(folder_name)
        print(f"Folder ID: {info['id']}")
        self.folder_id = info["id"]

    def _create_drive_service(self):
        """
        Create and return a Google Drive API service object.

        Returns:
            googleapiclient.discovery.Resource: A Google Drive API service object.
        """
        creds = Credentials.from_service_account_file(
            self.credentials_path, scopes=["https://www.googleapis.com/auth/drive.file"]
        )
        return build("drive", "v3", credentials=creds)

    def upload_photo(self, file_path: str) -> Optional[str]:
        """
        Upload a photo to the specified Google Drive folder.

        Args:
            file_path (str): Path to the photo file to be uploaded.

        Returns:
            Optional[str]: The file ID of the uploaded photo if successful, None otherwise.
        """
        try:
            file_name = os.path.basename(file_path)
            file_metadata = {"name": file_name, "parents": [self.folder_id]}
            media = MediaInMemoryUpload(open(file_path, "rb").read(), resumable=True)
            file = (
                self.drive_service.files()
                .create(body=file_metadata, media_body=media, fields="id", supportsAllDrives=True)
                .execute()
            )
            print(f'File uploaded successfully. File ID: {file.get("id")}')
            return file.get("id")
        except Exception as e:
            print(f"An error occurred while uploading the file: {e}")
            return None

    def upload_numpy_image(
        self, image: np.ndarray, file_name: str, format: str = "JPEG"
    ) -> Optional[str]:
        """
        Upload a NumPy image array to the specified Google Drive folder.

        Args:
            image (np.ndarray): NumPy array representing the image.
            file_name (str): Name to give the file in Google Drive.
            format (str): Image format (default is 'JPEG').

        Returns:
            Optional[str]: The file ID of the uploaded photo if successful, None otherwise.
        """
        try:
            # Convert NumPy array to PIL Image
            pil_image = Image.fromarray(image)

            # Save image to BytesIO object
            byte_arr = BytesIO()
            pil_image.save(byte_arr, format=format)
            byte_arr.seek(0)

            file_metadata = {"name": file_name, "parents": [self.folder_id]}
            media = MediaInMemoryUpload(
                byte_arr.getvalue(), mimetype=f"image/{format.lower()}", resumable=True
            )
            file = (
                self.drive_service.files()
                .create(body=file_metadata, media_body=media, fields="id", supportsAllDrives=True)
                .execute()
            )
            print(f'NumPy image uploaded successfully. File ID: {file.get("id")}')
            return file.get("id")
        except Exception as e:
            print(f"An error occurred while uploading the NumPy image: {e}")
            return None

    def create_folder(self, folder_name: str, parent_folder_id: Optional[str] = None) -> str:
        """
        Create a folder in Google Drive.

        Args:
            folder_name (str): Name of the folder to create.
            parent_folder_id (Optional[str]): ID of the parent folder. If None, creates in root.

        Returns:
            str: ID of the created folder.
        """
        folder_metadata: Dict[str, Any] = {
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder",
        }

        if parent_folder_id:
            folder_metadata["parents"] = [parent_folder_id]

        folder = (
            self.drive_service.files()
            .create(body=folder_metadata, fields="id", supportsAllDrives=True)
            .execute()
        )
        print(f'Folder created. ID: "{folder.get("id")}"')
        return folder.get("id")

    def folder_exists(self, folder_name: str) -> bool:
        """
        Check if a folder exists in Google Drive.

        Args:
            folder_id (str): ID of the folder to check.

        Returns:
            bool: True if the folder exists, False otherwise.
        """
        try:
            info = self.get_folder_by_name(folder_name)
            if info is not None:
                return True
        except Exception as e:
            print(e)
        return False

    def get_folder_by_name(
        self, folder_name: str, parent_folder_id: Optional[str] = None
    ) -> Optional[Dict[str, str]]:
        """
        Get a folder by its name in Google Drive.

        Args:
            folder_name (str): Name of the folder to search for.
            parent_folder_id (Optional[str]): ID of the parent folder to search in. If None, searches everywhere.

        Returns:
            Optional[Dict[str, str]]: Dictionary containing folder ID and name if found, None otherwise.
        """
        query = f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}'"
        if parent_folder_id:
            query += f" and '{parent_folder_id}' in parents"

        results = (
            self.drive_service.files()
            .list(
                q=query,
                spaces="drive",
                fields="files(id, name)",
                pageSize=1,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )

        items = results.get("files", [])

        if not items:
            print(f"No folder named '{folder_name}' found.")
            return None

        return {"id": items[0]["id"], "name": items[0]["name"]}

    def get_folder_info(self, folder_id: str) -> Optional[dict]:
        """
        Get information about a folder in Google Drive.

        Args:
            folder_id (str): ID of the folder to get information about.

        Returns:
            Optional[dict]: Dictionary containing folder information if it exists, None otherwise.
        """
        try:
            return (
                self.drive_service.files()
                .get(fileId=folder_id, fields="id, name, mimeType")
                .execute()
            )
        except:
            return None

    def upload_multiple_photos(
        self, files: List[Union[str, np.ndarray, Any]], file_names: Optional[List[str | Any]] = None
    ) -> list[Optional[str]]:
        """
        Upload multiple photos to the specified Google Drive folder.

        Args:
            files (list[Union[str, np.ndarray]]): List of file paths or NumPy arrays to be uploaded.
            file_names (Optional[list[str]]): List of file names for NumPy arrays. Required if any NumPy arrays are provided.

        Returns:
            list[Optional[str]]: List of file IDs for the uploaded photos.
                                 None for any photos that failed to upload.
        """
        results = []
        for i, file in enumerate(files):
            if isinstance(file, str):
                results.append(self.upload_photo(file))
            elif isinstance(file, np.ndarray):
                if file_names is None or len(file_names) <= i:
                    raise ValueError("File names must be provided for NumPy arrays")
                results.append(self.upload_numpy_image(file, file_names[i]))
            else:
                print(f"Unsupported file type at index {i}")
                results.append(None)
        return results

    def get_folder_contents(self, folder_id: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Get all contents (files and subfolders) of a specified folder.

        Args:
            folder_id (str): ID of the folder to get contents from.

        Returns:
            List[Dict[str, str]]: List of dictionaries containing id, name, and mimeType of each item.
        """
        if folder_id is None:
            folder_id = self.folder_id
        query = f"'{folder_id}' in parents"
        results = []
        page_token = None

        while True:
            response = (
                self.drive_service.files()
                .list(
                    q=query,
                    spaces="drive",
                    fields="nextPageToken, files(id, name, mimeType)",
                    pageToken=page_token,
                )
                .execute()
            )

            results.extend(response.get("files", []))
            page_token = response.get("nextPageToken")

            if not page_token:
                break

        return results

    def get_web_link(self, file_id: str) -> Optional[str]:
        """
        Get the web link (URL) for a file or folder in Google Drive.

        Args:
            file_id (str): ID of the file or folder.

        Returns:
            Optional[str]: Web link of the file or folder, or None if not found.
        """
        try:
            file = self.drive_service.files().get(fileId=file_id, fields="webViewLink").execute()
            return file.get("webViewLink")
        except Exception as e:
            print(f"Error getting web link: {e}")
            return None

    def transfer_ownership(self, file_id: str, new_owner_email: str) -> bool:
        """
        Transfer ownership of a file to another user.

        Args:
            file_id (str): ID of the file to transfer ownership.
            new_owner_email (str): Email address of the new owner.

        Returns:
            bool: True if ownership transfer was successful, False otherwise.
        """
        try:
            # First, share the file with the new owner if not already shared
            self.drive_service.permissions().create(
                fileId=file_id,
                body={"type": "user", "role": "owner", "emailAddress": new_owner_email},
                transferOwnership=True,
            ).execute()

            print(f"Ownership transferred to {new_owner_email} for file ID: {file_id}")
            return True
        except Exception as e:
            print(f"Error transferring ownership: {e}")
            return False

    def transfer_ownership_of_folder_contents(
        self, new_owner_email: str, folder_id: Optional[str] = None
    ):
        """
        Transfer ownership of all files in a folder to a new owner.

        Args:
            folder_id (str): ID of the folder containing the files.
            new_owner_email (str): Email address of the new owner.
        """
        if folder_id is None:
            folder_id = self.folder_id
        contents = self.get_folder_contents(folder_id)
        for item in contents:
            if item["mimeType"] != "application/vnd.google-apps.folder":
                success = self.transfer_ownership(item["id"], new_owner_email)
                if success:
                    print(f"Transferred ownership of '{item['name']}' to {new_owner_email}")
                else:
                    print(f"Failed to transfer ownership of '{item['name']}'")

    def download_folder_contents(self, local_path: str, folder_id: Optional[str] = None):
        """
        Download all contents of a specified folder to a local directory.

        Args:
            folder_id (str): ID of the folder to download contents from.
            local_path (str): Local path to save the downloaded contents.
        """

        if folder_id is None:
            folder_id = self.folder_id

        if not os.path.exists(local_path):
            os.makedirs(local_path)

        items = self.get_folder_contents(folder_id)

        for item in items:
            item_name = item["name"]
            item_id = item["id"]
            item_type = item["mimeType"]

            local_item_path = os.path.join(local_path, item_name)

            if item_type == "application/vnd.google-apps.folder":
                # If it's a folder, create it locally and recurse
                os.makedirs(local_item_path, exist_ok=True)
                self.download_folder_contents(item_id, local_item_path)
            else:
                # If it's a file, download it
                self.download_file(item_id, local_item_path)

    def download_file(self, file_id: str, local_path: str):
        """
        Download a single file from Google Drive.

        Args:
            file_id (str): ID of the file to download.
            local_path (str): Local path to save the downloaded file.
        """
        try:
            request = self.drive_service.files().get_media(fileId=file_id)
            file = io.BytesIO()
            downloader = MediaIoBaseDownload(file, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                print(f"Download {int(status.progress() * 100)}%.")

            file.seek(0)
            with open(local_path, "wb") as f:
                f.write(file.read())
            print(f"File downloaded: {local_path}")
        except Exception as e:
            print(f"Error downloading file {file_id}: {e}")

    def delete_folder(self, folder_id: str, permanent: bool = False) -> Optional[bool]:
        """
        Delete a folder from Google Drive.

        This function attempts to delete a folder with the given ID from Google Drive.
        It can either permanently delete the folder or move it to trash.

        Args:
            folder_id (str): The ID of the folder to be deleted.
            permanent (bool, optional): If True, permanently delete the folder.
                                        If False, move to trash. Defaults to True.

        Returns:
            Optional[bool]: True if deletion was successful, False if moved to trash,
                            None if an error occurred.

        Raises:
            HttpError: If an HTTP error occurs during the API call.

        Example:
            >>> result = delete_folder("1234567890abcdef")
            >>> if result is True:
            ...     print("Folder deleted permanently.")
            >>> elif result is False:
            ...     print("Folder moved to trash.")
            >>> else:
            ...     print("An error occurred.")
        """
        try:
            if permanent:
                self.drive_service.files().delete(
                    fileId=folder_id, supportsAllDrives=True
                ).execute()
                print(f"Folder with ID {folder_id} deleted permanently.")
                return True
            else:
                file_metadata = {"trashed": True}
                self.drive_service.files().update(
                    fileId=folder_id, body=file_metadata, supportsAllDrives=True
                ).execute()
                print(f"Folder with ID {folder_id} moved to trash.")
                return False
        except HttpError as e:
            if e.resp.status == 404:
                print(f"Folder with ID {folder_id} not found.")
            else:
                print(f"An HTTP error occurred: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None


# Example usage
if __name__ == "__main__":
    uploader = GoogleDriveUploader()

    # Upload a single photo from file
    uploader.upload_photo("receptacle.png")

    # Upload a NumPy image
    import numpy as np

    numpy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    uploader.upload_numpy_image(numpy_image, "random_image.jpg")

    # Upload multiple photos (mix of file paths and NumPy arrays)
    photo_files = [
        "object.png",
        np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8),
    ]
    file_names = [None, "numpy_image.png", None]
    uploader.upload_multiple_photos(photo_files, file_names)

    # uploader.transfer_ownership_of_folder_contents("cpaxton@hello-robot.com")

    contents = uploader.get_folder_contents()
    for obj in contents:
        print(f"- {obj['name']} ({obj['mimeType']}) id = {obj['id']}")
        link = uploader.get_web_link(obj["id"])
        print("  Web link:", link)

    uploader.download_folder_contents(local_path="downloads")

    # TODO: do you really want to do this?
    # uploader.delete_folder(uploader.folder_id, permanent=False)
