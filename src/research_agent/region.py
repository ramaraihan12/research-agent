"""Regions where the live model part of the public demo is switched off.

The demo runs on Gemini's free tier, and Google's terms for it don't allow serving users in the
EEA, Switzerland or the UK. The check reads the country Vercel puts on each request. Nothing
here logs or stores the country, the IP or any header.
"""

from fastapi import HTTPException, Request

BLOCKED_COUNTRIES = frozenset(
    {
        # EU
        *("AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE"),
        *("IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE"),
        # EEA, Switzerland, UK
        *("IS", "LI", "NO", "CH", "GB"),
        # EU outermost regions that geo databases report under their own code
        *("AX", "GF", "GP", "MQ", "RE", "YT", "MF"),
    }
)

COUNTRY_HEADER = "x-vercel-ip-country"

BLOCKED_MESSAGE = (
    "The live AI part of this demo isn't available in the EEA, Switzerland or the UK, because "
    "it runs on a free model tier whose terms exclude those regions."
)


def is_blocked(request: Request) -> bool:
    # Without the header (local dev, tests, any host other than Vercel) nobody is blocked.
    country = request.headers.get(COUNTRY_HEADER, "").strip().upper()
    return country in BLOCKED_COUNTRIES


async def require_allowed_region(request: Request) -> None:
    """Dependency for every route that can reach the model."""
    if is_blocked(request):
        raise HTTPException(451, BLOCKED_MESSAGE)
