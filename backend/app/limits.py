"""Bound total request size before multipart/JSON parsing, including chunked bodies."""
from starlette.responses import JSONResponse


class BodyLimitMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['method'] not in ('POST', 'PUT', 'PATCH'):
            return await self.app(scope, receive, send)
        limit = 512_000 if scope['path'] == '/v1/profile' else 16 * 1024 * 1024
        parts, length = [], 0
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                return
            part = message.get('body', b'')
            length += len(part)
            if length > limit:
                return await JSONResponse({'detail': 'Request too large'}, status_code=413)(scope, receive, send)
            parts.append(part)
            if not message.get('more_body', False):
                break
        body = b''.join(parts)
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {'type': 'http.request', 'body': body, 'more_body': False}
            return await receive()

        return await self.app(scope, replay, send)
