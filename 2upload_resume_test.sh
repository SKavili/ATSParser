#!/bin/bash
# Script to test resume upload with multiple resume files from a folder

# Resume folder path
RESUME_FOLDER="/home/ananya-k/Documents/email"
API_URL="http://localhost:8000/api/v1/upload-resume"

# Function to process a single file
process_file() {
    local RESUME_FILE="$1"
    local EXTRACT_MODULES="$2"
    local FILE_NUM="$3"
    local FILENAME=$(basename "$RESUME_FILE")
    
    echo ""
    echo "========================================"
    echo "[${FILE_NUM}] Processing: ${FILENAME}"
    echo "========================================"
    echo ""
    echo "[INFO] This file will be processed ONE BY ONE"
    echo "[INFO] Selected modules will be extracted using OLLAMA"
    echo "[INFO] Results will be saved to database"
    echo ""
    
    # Extract candidate name from filename (remove extension, replace underscores/hyphens with spaces)
    CANDIDATE_NAME="${FILENAME}"
    CANDIDATE_NAME="${CANDIDATE_NAME%.pdf}"
    CANDIDATE_NAME="${CANDIDATE_NAME%.doc}"
    CANDIDATE_NAME="${CANDIDATE_NAME%.docx}"
    CANDIDATE_NAME="${CANDIDATE_NAME%.txt}"
    CANDIDATE_NAME="${CANDIDATE_NAME//_/ }"
    CANDIDATE_NAME="${CANDIDATE_NAME//-/ }"
    
    # Job role is now automatically extracted from resume using OLLAMA
    # You can optionally provide a job_role hint, but it will be extracted automatically
    # Set to empty to let the system extract it automatically
    JOB_ROLE=""
    
    echo "[INFO] Candidate Name: ${CANDIDATE_NAME}"
    echo "[INFO] Job Role: (will be extracted automatically from resume)"
    echo "[INFO] Modules to extract: ${EXTRACT_MODULES}"
    echo ""
    
    # Upload the file - this triggers extraction based on selected modules
    # Note: If "role" is in extract_modules, job role will be extracted automatically
    echo "[INFO] Uploading to API and extracting selected modules..."
    if [ -z "$JOB_ROLE" ]; then
        # No job_role provided - let system extract it automatically
        curl -X POST "${API_URL}" \
          -H "accept: application/json" \
          -H "Content-Type: multipart/form-data" \
          -F "file=@${RESUME_FILE}" \
          -F "candidate_name=${CANDIDATE_NAME}" \
          -F "extract_modules=${EXTRACT_MODULES}"
    else
        # Optional job_role hint provided (will still be extracted if "role" module is selected)
        curl -X POST "${API_URL}" \
          -H "accept: application/json" \
          -H "Content-Type: multipart/form-data" \
          -F "file=@${RESUME_FILE}" \
          -F "candidate_name=${CANDIDATE_NAME}" \
          -F "job_role=${JOB_ROLE}" \
          -F "extract_modules=${EXTRACT_MODULES}"
    fi
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "[ERROR] Failed to upload: ${FILENAME}"
        return 1
    else
        echo ""
        echo "[SUCCESS] Uploaded and processed: ${FILENAME}"
        echo "[INFO] Extraction completed for selected modules (check server logs for details)"
        return 0
    fi
}

echo "========================================"
echo "  Testing Resume Upload (Multiple Files)"
echo "========================================"
echo ""
echo "Resume Folder: ${RESUME_FOLDER}"
echo "API URL: ${API_URL}"
echo ""
echo "========================================"
echo "  MODULE SELECTION MENU"
echo "========================================"
echo ""
echo "Select which modules to extract:"
echo ""
echo "  0 - Extract ALL modules (default)"
echo "  1 - Designation only"
echo "  2 - Name only"
echo "  3 - Role only"
echo "  4 - Email only"
echo "  5 - Mobile only"
echo "  6 - Experience only"
echo "  7 - Domain only"
echo "  8 - Education only"
echo "  9 - Skills only"
echo ""
echo "  You can also select multiple modules:"
echo "  Example: 1,2,3 or 1,9 or designation,skills,role"
echo ""
read -p "Enter your choice (0-9, comma-separated, or 'all'): " MODULE_CHOICE

# Default to "all" if empty
if [ -z "$MODULE_CHOICE" ]; then
    MODULE_CHOICE="0"
fi

# Convert choice to extract_modules format
if [ "$MODULE_CHOICE" = "0" ]; then
    EXTRACT_MODULES="all"
else
    EXTRACT_MODULES="$MODULE_CHOICE"
fi

echo ""
echo "[INFO] Selected modules: ${EXTRACT_MODULES}"
echo ""

# Check if resume folder exists
if [ ! -d "$RESUME_FOLDER" ]; then
    echo "[ERROR] Resume folder not found at: ${RESUME_FOLDER}"
    echo "Please check the folder path."
    read -p "Press Enter to exit..."
    exit 1
fi

# Check if curl is available
if ! command -v curl &> /dev/null; then
    echo "[ERROR] curl is not installed or not in PATH."
    echo "Please install curl: sudo apt-get install curl"
    echo ""
    read -p "Press Enter to exit..."
    exit 1
fi

echo "[INFO] Scanning folder for resume files..."
echo ""

# Counter for processed files
FILE_COUNT=0
SUCCESS_COUNT=0
FAILED_COUNT=0

# Process PDF files
for file in "${RESUME_FOLDER}"/*.pdf "${RESUME_FOLDER}"/*.PDF; do
    if [ -f "$file" ]; then
        ((FILE_COUNT++))
        process_file "$file" "$EXTRACT_MODULES" "$FILE_COUNT"
        if [ $? -eq 0 ]; then
            ((SUCCESS_COUNT++))
        else
            ((FAILED_COUNT++))
        fi
        echo ""
        echo "Waiting 2 seconds before processing next file..."
        sleep 2
    fi
done

# Process DOC files
for file in "${RESUME_FOLDER}"/*.doc "${RESUME_FOLDER}"/*.DOC; do
    if [ -f "$file" ]; then
        ((FILE_COUNT++))
        process_file "$file" "$EXTRACT_MODULES" "$FILE_COUNT"
        if [ $? -eq 0 ]; then
            ((SUCCESS_COUNT++))
        else
            ((FAILED_COUNT++))
        fi
        echo ""
        echo "Waiting 2 seconds before processing next file..."
        sleep 2
    fi
done

# Process DOCX files
for file in "${RESUME_FOLDER}"/*.docx "${RESUME_FOLDER}"/*.DOCX; do
    if [ -f "$file" ]; then
        ((FILE_COUNT++))
        process_file "$file" "$EXTRACT_MODULES" "$FILE_COUNT"
        if [ $? -eq 0 ]; then
            ((SUCCESS_COUNT++))
        else
            ((FAILED_COUNT++))
        fi
        echo ""
        echo "Waiting 2 seconds before processing next file..."
        sleep 2
    fi
done

# Note: Only processing .pdf, .doc, .docx files (not .txt)

echo ""
echo "========================================"
echo "  Upload Summary"
echo "========================================"
echo "Total files found: ${FILE_COUNT}"
echo "Successfully uploaded: ${SUCCESS_COUNT}"
echo "Failed: ${FAILED_COUNT}"
echo "========================================"
read -p "Press Enter to exit..."
exit 0

