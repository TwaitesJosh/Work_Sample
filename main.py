from typing import Union
import backend.job as job



def like_jobs(job_id: int):
    my_job = job.Job(job_id)
    my_job_similarity = my_job.job_similarity()
