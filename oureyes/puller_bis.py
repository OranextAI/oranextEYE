from oureyes.stream_manager import get_stream

async def pull_stream(cam_name):
    """
    Returns the asyncio.Queue that asynchronously yields frames for the given camera.
    """
    return await get_stream(cam_name)
