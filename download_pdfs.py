#!/usr/bin/env python3
"""
Download financial PDFs to the inbox directory for auto-ingestion.
"""

import os
import requests
from urllib.parse import urlparse
import time

# List of PDF URLs to download
url_list = [
    'https://www.berkshirehathaway.com/2021ar/2021ar.pdf',
    'https://www.berkshirehathaway.com/2022ar/2022ar.pdf', 
    'https://www.berkshirehathaway.com/2023ar/2023ar.pdf',
    'https://s26.q4cdn.com/747928648/files/doc_financials/2021/q4/d13acb37-2f7e-411d-8ed1-6516668bf861.pdf',
    'https://s26.q4cdn.com/747928648/files/doc_financials/2022/q4/AMEX-10K-4Q22.pdf',
    'https://s26.q4cdn.com/747928648/files/doc_financials/2023/q4/American-Express-Company-10K-4Q2024.pdf',
    'https://abc.xyz/assets/43/44/675b83d7455885c4615d848d52a4/goog-10-k-2023.pdf',
    'https://abc.xyz/assets/9a/bd/838c917c4b4ab21f94e84c3c2c65/goog-10-k-q4-2022.pdf',
    'https://abc.xyz/assets/d9/85/b7649a9f48c4960adbce5bd9fb54/20220202-alphabet-10k.pdf',
]

def get_filename_from_url(url):
    """Extract filename from URL, with fallbacks for better naming."""
    parsed = urlparse(url)
    filename = os.path.basename(parsed.path)
    
    # If no filename in path, generate one from the URL
    if not filename or not filename.endswith('.pdf'):
        if 'berkshirehathaway' in url:
            if '2021ar' in url:
                filename = 'berkshire-hathaway-2021-annual-report.pdf'
            elif '2022ar' in url:
                filename = 'berkshire-hathaway-2022-annual-report.pdf'
            elif '2023ar' in url:
                filename = 'berkshire-hathaway-2023-annual-report.pdf'
        elif 'q4cdn.com' in url and 'amex' in url.lower():
            if '2021' in url:
                filename = 'american-express-10k-2021.pdf'
            elif '2022' in url:
                filename = 'american-express-10k-2022.pdf'
            elif '2023' in url:
                filename = 'american-express-10k-2023.pdf'
        elif 'abc.xyz' in url:
            if 'goog-10-k-2023' in url:
                filename = 'alphabet-10k-2023.pdf'
            elif 'goog-10-k-q4-2022' in url:
                filename = 'alphabet-10k-q4-2022.pdf'
            elif '20220202-alphabet-10k' in url:
                filename = 'alphabet-10k-2021.pdf'
        
        # Fallback: use the original filename from URL
        if not filename:
            filename = os.path.basename(parsed.path) or 'downloaded.pdf'
    
    return filename

def download_pdf(url, dest_dir):
    """Download a PDF from URL to destination directory."""
    try:
        print(f"Downloading: {url}")
        
        # Get filename
        filename = get_filename_from_url(url)
        dest_path = os.path.join(dest_dir, filename)
        
        # Download with stream=True for large files
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, stream=True, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Check if it's actually a PDF
        content_type = response.headers.get('content-type', '')
        if 'pdf' not in content_type.lower():
            print(f"Warning: {url} doesn't appear to be a PDF (content-type: {content_type})")
        
        # Write to file
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        file_size = os.path.getsize(dest_path)
        print(f"✅ Downloaded: {filename} ({file_size:,} bytes)")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to download {url}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")
        return False

def main():
    # Create inbox directory if it doesn't exist
    inbox_dir = 'data/inbox'
    os.makedirs(inbox_dir, exist_ok=True)
    
    print(f"Downloading {len(url_list)} PDFs to {inbox_dir}/")
    print("=" * 50)
    
    successful = 0
    failed = 0
    
    for i, url in enumerate(url_list, 1):
        print(f"\n[{i}/{len(url_list)}]", end=" ")
        
        if download_pdf(url, inbox_dir):
            successful += 1
        else:
            failed += 1
        
        # Small delay between downloads to be respectful
        if i < len(url_list):
            time.sleep(1)
    
    print("\n" + "=" * 50)
    print(f"Download complete: {successful} successful, {failed} failed")
    
    # List files in inbox
    files = [f for f in os.listdir(inbox_dir) if f.endswith('.pdf')]
    print(f"\nFiles in inbox ({len(files)} PDFs):")
    for f in sorted(files):
        size = os.path.getsize(os.path.join(inbox_dir, f))
        print(f"  {f} ({size:,} bytes)")

if __name__ == '__main__':
    main()