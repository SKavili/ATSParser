#!/bin/bash
# Sample cURL requests for ATS Backend API

BASE_URL="http://localhost:8000/api/v1"

echo "[INFO] Testing ATS Backend API"
echo "=============================="

# Health check
echo -e "\n[1] Health Check"
curl -X GET "${BASE_URL}/health" \
  -H "accept: application/json"

# Upload Resume
echo -e "\n\n[2] Upload Resume"
curl -X POST "${BASE_URL}/upload-resume" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_resume.pdf" \
  -F "candidate_name=John Doe" \
  -F "job_role=Software Engineer"

# Create Job
echo -e "\n\n[3] Create Job"
curl -X POST "${BASE_URL}/create-job" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Senior Software Engineer",
    "description": "We are looking for an experienced software engineer with Python and FastAPI experience. The ideal candidate should have 5+ years of experience in backend development.",
    "required_skills": ["Python", "FastAPI", "MySQL", "Docker"],
    "location": "San Francisco, CA"
  }'

# Match Job (by description)
echo -e "\n\n[4] Match Job by Description"
curl -X POST "${BASE_URL}/match" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "job_description": "We are looking for an experienced software engineer with Python and FastAPI experience.",
    "top_k": 5
  }'

# Match Job (by job_id) - Replace JOB_ID with actual job_id from create-job response
echo -e "\n\n[5] Match Job by Job ID"
echo "Note: Replace JOB_ID with actual job_id from create-job response"
# curl -X POST "${BASE_URL}/match" \
#   -H "accept: application/json" \
#   -H "Content-Type: application/json" \
#   -d '{
#     "job_id": "JOB_ID",
#     "top_k": 5
#   }'

echo -e "\n\n[DONE] Sample requests completed"

