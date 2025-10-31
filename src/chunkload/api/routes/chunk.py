from fastapi import APIRouter

from chunkload.building_blocks.chunk import Chunk

router = APIRouter()

@router.get("/chunk/graphics", response_model=dict)
def get_chunk_graphics():
    """
    Return the H→shape and T→color mappings from the Chunk class.
    """
    return {
        "H_SHAPE_MAP": Chunk.H_SHAPE_MAP,
        "T_COLOR_MAP": Chunk.T_COLOR_MAP,
    }
