from fastapi import APIRouter, Request


router = APIRouter()


@router.get("/documents")
def list_documents(request: Request):

    # Which PDFs are indexed (read from the docstore, so it survives restarts)
    return request.app.state.components.docstore.list_sources()
