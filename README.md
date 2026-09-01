# ai-service
A unified AI service. Serves models for different kind of computer vision tasks. Integrates standard endpoint calls and jobs to be picked up by celery workers. To be used with the iquana-tool.

## Configuration pushed in from the backend

Credentials this service needs but does not own are edited on the backend's admin
page (Datasets → Admin → Settings) and sent here over `PATCH /config`. Today that
is the Hugging Face access token, which decides whether gated model weights
(SAM, DINOv3) can be downloaded.

Pushed values are held **in memory only**: restarting this service drops them and
it falls back to its own `HF_ACCESS_TOKEN`. The settings page shows what this
service is currently holding, so the drift is visible, with a *Send to AI
service* button to push it again.

`GET /config` reports whether a token is set and its last four characters. The
value itself is never returned.

### `AI_SERVICE_ADMIN_TOKEN`

Optional shared secret. When set, `/config` requires the same value in an
`X-Admin-Token` header and the backend must be given the matching
`AI_SERVICE_ADMIN_TOKEN`. When unset the endpoint is open, matching the rest of
this service, which has no authentication and is expected to be reachable only
from the backend.

**Set it on any deployment where this service's port is reachable from anywhere
else** — without it, whoever can reach the port can replace the token this
service authenticates to Hugging Face with.
