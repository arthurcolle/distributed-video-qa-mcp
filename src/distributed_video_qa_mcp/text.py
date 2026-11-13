"""Shared text helpers."""

from __future__ import annotations

# Function words and filler that carry no topic. Used to keep keyword search and
# chapter keywords from matching on "what", "the" or "you know".
STOPWORDS = frozenset(
    """a about after again all also am an and any are as at be because been before being but by can could
    did do does doing down for from get go going got had has have he her here him his how i if in into is
    it its just know like me more most my no not now of on one only or other our out over really right
    so some than that the their them then there these they thing things think this those to too up us
    very was we well were what when where which who why will with would yeah you your
    i'm it's that's they're we're you're don't didn't can't won't isn't there's here's let's i've you've
    i'd i'll you'll we'll he's she's what's want say said see look make made new way lot even still much
    many every each another going gonna actually basically""".split()
)
