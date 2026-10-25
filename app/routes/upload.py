from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.services import actions


router = APIRouter()

UPLOAD_DIR = "data/raw"


@router.post("/upload")
def upload_pdf(
    request: Request,
    file: UploadFile = File(...)
):

    try:
        return actions.upload_pdf(
            request.app.state.components,
            file.filename,
            file.file,
            upload_dir=UPLOAD_DIR
        )

    except actions.ActionError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
