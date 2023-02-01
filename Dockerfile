# 
FROM python:3.9

# 
WORKDIR /code

# 
COPY ./backend /code/backend

# 
COPY ./requirements.txt /code/requirements.txt
COPY occupations_en.csv  /backend/occupations_en.csv
COPY Occs_processed.csv  /backend/Occs_processed.csv
COPY skills_en.csv  /backend/skills_en.csv
COPY skills_en.csv  /backend/skills_en.csv
COPY BERT.npy  /backend/BERT.npy
# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt



# 
CMD ["uvicorn", "backend.main:Mmin", "--host", "0.0.0.0", "--port", "80"]