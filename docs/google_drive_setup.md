# Creating Google Drive JSON Credentials

Warning: this is experimental code, and may not be all that useful. It's designed to provide a way to upload data from a Stretch to Google Cloud services.

First, you need to install the google cloud python API:

```bash
python -m pip install --upgrade google-api-python-client
```

Follow these steps to create a JSON file for Google Drive credentials:

## Step 1: Go to Google Cloud Console

- Visit the [Google Cloud Console](https://console.cloud.google.com/).

## Step 2: Create a New Project

- Click on the project dropdown in the top navigation bar.
- Select **New Project** or choose an existing project.

## Step 3: Enable Google Drive API

1. In the sidebar, click on **APIs & Services** > **Library**.
1. Search for **Google Drive API**.
1. Click on it and then click the **Enable** button.

## Step 4: Create a Service Account

1. In the sidebar, click on **APIs & Services** > **Credentials**.
1. Click on **Create Credentials** and select **Service Account**.
1. Fill in the required information (Service Account Name, etc.) and click **Create**.

## Step 5: Generate a JSON Key for the Service Account

1. In the service account list, find the one you just created.
1. Click on the service account name to view its details.
1. Go to the **Keys** tab.
1. Click **Add Key** > **Create new key**.
1. Choose **JSON** as the key type and click **Create**.
1. The JSON key file will be automatically downloaded to your computer.

## Step 6: Share Google Drive Folder with Service Account (Optional)

- If you need to access a specific Google Drive folder, share that folder with the service account's email address (found in the JSON key file).

## Step 7: Use the JSON Key in Your Code

In your Python code, use the path to the downloaded JSON file as the `credentials_path`:

```python
from google.oauth2.service_account import Credentials

creds = Credentials.from_service_account_file(
    'path/to/your/downloaded/service-account-key.json', 
    scopes=['https://www.googleapis.com/auth/drive.file']
)
```

For our purposes, copy the downloaded credentials to `$HOME/.stretch/credentials.json`:

```bash
mkdir -p $HOME/.stretch
cp path/to/your/downloaded/service-account-key.json $HOME/.stretch/credentials.json
```

Note that this will allow you to access the Google Drive API with the service account's permissions. Be careful with this file!
