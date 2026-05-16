"""Canonical Chicago corridors that drive the synthetic-route corpus.

A *corridor* is one direction of one origin-destination pair on one mode.
We always seed both directions of every OD (inbound + outbound) so the
directional asymmetry is preserved -- DOOR_TO_DOOR.md is explicit that
two corridors per OD is the rule, never one bidirectional record.

The collector cycles through corridors on a fixed cadence rather than
sampling random trips. Each corridor produces one synthetic prediction
per ``cadence_seconds``; that prediction is later graded against the
recorded feed stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import duckdb


@dataclass(frozen=True)
class Corridor:
    corridor_id: str
    mode: str                    # 'L' | 'bus' | 'metra' | 'intercampus'
    line: str                    # API line code (e.g. 'Red'), bus route ('22'), Metra route_id ('UP-N')
    direction: str               # 'inbound' | 'outbound' | 'northbound' | 'southbound' | 'eastbound' | 'westbound'
    origin_label: str
    origin_latitude: float
    origin_longitude: float
    destination_label: str
    destination_latitude: float
    destination_longitude: float
    boarding_int_id: int         # 0 when not applicable (non-L modes)
    boarding_text_id: str | None
    alighting_int_id: int
    alighting_text_id: str | None
    schedule_headway_seconds: float
    cadence_seconds: float = 300.0
    priority: int = 5


# Seed corridors. Two rows per OD: ``-ib`` is the toward-Loop / toward-Evanston
# / canonical-A direction, ``-ob`` is the reverse. Intercampus is a one-way
# loop in each direction, so each direction is its own corridor.
SEED_CORRIDORS: tuple[Corridor, ...] = (
    # --- Metra UP-N: Evanston (Davis) <-> Ogilvie ---
    Corridor(
        corridor_id="metra-upn-evanston-otc-ib",
        mode="metra", line="UP-N", direction="inbound",
        origin_label="Evanston (Davis St.)", origin_latitude=42.0467, origin_longitude=-87.6837,
        destination_label="Chicago OTC", destination_latitude=41.8855, destination_longitude=-87.6406,
        boarding_int_id=0, boarding_text_id="EVANSTON",
        alighting_int_id=0, alighting_text_id="OTC",
        schedule_headway_seconds=1800.0, cadence_seconds=300.0, priority=1,
    ),
    Corridor(
        corridor_id="metra-upn-evanston-otc-ob",
        mode="metra", line="UP-N", direction="outbound",
        origin_label="Chicago OTC", origin_latitude=41.8855, origin_longitude=-87.6406,
        destination_label="Evanston (Davis St.)", destination_latitude=42.0467, destination_longitude=-87.6837,
        boarding_int_id=0, boarding_text_id="OTC",
        alighting_int_id=0, alighting_text_id="EVANSTON",
        schedule_headway_seconds=1800.0, cadence_seconds=300.0, priority=1,
    ),
    Corridor(
        corridor_id="metra-upn-central-otc-ib",
        mode="metra", line="UP-N", direction="inbound",
        origin_label="Central St.", origin_latitude=42.0613, origin_longitude=-87.6929,
        destination_label="Chicago OTC", destination_latitude=41.8855, destination_longitude=-87.6406,
        boarding_int_id=0, boarding_text_id="CENTRALST",
        alighting_int_id=0, alighting_text_id="OTC",
        schedule_headway_seconds=1800.0, cadence_seconds=300.0, priority=2,
    ),
    Corridor(
        corridor_id="metra-upn-central-otc-ob",
        mode="metra", line="UP-N", direction="outbound",
        origin_label="Chicago OTC", origin_latitude=41.8855, origin_longitude=-87.6406,
        destination_label="Central St.", destination_latitude=42.0613, destination_longitude=-87.6929,
        boarding_int_id=0, boarding_text_id="OTC",
        alighting_int_id=0, alighting_text_id="CENTRALST",
        schedule_headway_seconds=1800.0, cadence_seconds=300.0, priority=2,
    ),

    # --- Intercampus loops ---
    Corridor(
        corridor_id="intercampus-central-loyola-sb",
        mode="intercampus", line="intercampus", direction="southbound",
        origin_label="Central/Jackson (IB)", origin_latitude=42.0613, origin_longitude=-87.6943,
        destination_label="Sheridan/Loyola (IB)", destination_latitude=41.9999, destination_longitude=-87.6595,
        boarding_int_id=0, boarding_text_id="b3f50cbe-621f-4664-934a-fe48d4901250",
        alighting_int_id=0, alighting_text_id="e5aa8b6f-44b5-4c4b-becd-1125a1fa4db4",
        schedule_headway_seconds=900.0, cadence_seconds=300.0, priority=3,
    ),
    Corridor(
        corridor_id="intercampus-loyola-central-nb",
        mode="intercampus", line="intercampus", direction="northbound",
        origin_label="Sheridan/Loyola (OB)", origin_latitude=41.9999, origin_longitude=-87.6595,
        destination_label="Central/Jackson (OB)", destination_latitude=42.0613, destination_longitude=-87.6943,
        boarding_int_id=0, boarding_text_id="e647afb1-e56d-4b28-b58a-b581f27b3e90",
        alighting_int_id=0, alighting_text_id="c28a43f2-95c6-442f-9077-adfcdda4a4cf",
        schedule_headway_seconds=900.0, cadence_seconds=300.0, priority=3,
    ),

    # --- CTA Red Line: Belmont <-> Lake (North leg) ---
    Corridor(
        corridor_id="cta-red-belmont-lake-sb",
        mode="L", line="Red", direction="southbound",
        origin_label="Belmont", origin_latitude=41.9398, origin_longitude=-87.6531,
        destination_label="Lake", destination_latitude=41.8848, destination_longitude=-87.6280,
        boarding_int_id=41320, boarding_text_id=None,
        alighting_int_id=41660, alighting_text_id=None,
        schedule_headway_seconds=540.0, cadence_seconds=300.0, priority=2,
    ),
    Corridor(
        corridor_id="cta-red-belmont-lake-nb",
        mode="L", line="Red", direction="northbound",
        origin_label="Lake", origin_latitude=41.8848, origin_longitude=-87.6280,
        destination_label="Belmont", destination_latitude=41.9398, destination_longitude=-87.6531,
        boarding_int_id=41660, boarding_text_id=None,
        alighting_int_id=41320, alighting_text_id=None,
        schedule_headway_seconds=540.0, cadence_seconds=300.0, priority=2,
    ),

    # --- CTA Red Line: 95th/Dan Ryan <-> Lake (South leg) ---
    Corridor(
        corridor_id="cta-red-95th-lake-nb",
        mode="L", line="Red", direction="northbound",
        origin_label="95th/Dan Ryan", origin_latitude=41.7224, origin_longitude=-87.6244,
        destination_label="Lake", destination_latitude=41.8848, destination_longitude=-87.6280,
        boarding_int_id=40450, boarding_text_id=None,
        alighting_int_id=41660, alighting_text_id=None,
        schedule_headway_seconds=540.0, cadence_seconds=300.0, priority=4,
    ),
    Corridor(
        corridor_id="cta-red-95th-lake-sb",
        mode="L", line="Red", direction="southbound",
        origin_label="Lake", origin_latitude=41.8848, origin_longitude=-87.6280,
        destination_label="95th/Dan Ryan", destination_latitude=41.7224, destination_longitude=-87.6244,
        boarding_int_id=41660, boarding_text_id=None,
        alighting_int_id=40450, alighting_text_id=None,
        schedule_headway_seconds=540.0, cadence_seconds=300.0, priority=4,
    ),

    # --- CTA Blue Line: O'Hare <-> Clark/Lake ---
    Corridor(
        corridor_id="cta-blue-ohare-loop-eb",
        mode="L", line="Blue", direction="eastbound",
        origin_label="O'Hare", origin_latitude=41.9777, origin_longitude=-87.9042,
        destination_label="Clark/Lake", destination_latitude=41.8857, destination_longitude=-87.6309,
        boarding_int_id=40890, boarding_text_id=None,
        alighting_int_id=40380, alighting_text_id=None,
        schedule_headway_seconds=600.0, cadence_seconds=300.0, priority=3,
    ),
    Corridor(
        corridor_id="cta-blue-ohare-loop-wb",
        mode="L", line="Blue", direction="westbound",
        origin_label="Clark/Lake", origin_latitude=41.8857, origin_longitude=-87.6309,
        destination_label="O'Hare", destination_latitude=41.9777, destination_longitude=-87.9042,
        boarding_int_id=40380, boarding_text_id=None,
        alighting_int_id=40890, alighting_text_id=None,
        schedule_headway_seconds=600.0, cadence_seconds=300.0, priority=3,
    ),

    # --- CTA Pink Line: 54th/Cermak <-> Clark/Lake ---
    Corridor(
        corridor_id="cta-pink-cermak-loop-eb",
        mode="L", line="Pink", direction="eastbound",
        origin_label="54th/Cermak", origin_latitude=41.8518, origin_longitude=-87.7567,
        destination_label="Clark/Lake", destination_latitude=41.8857, destination_longitude=-87.6309,
        boarding_int_id=40580, boarding_text_id=None,
        alighting_int_id=40380, alighting_text_id=None,
        schedule_headway_seconds=900.0, cadence_seconds=300.0, priority=4,
    ),
    Corridor(
        corridor_id="cta-pink-cermak-loop-wb",
        mode="L", line="Pink", direction="westbound",
        origin_label="Clark/Lake", origin_latitude=41.8857, origin_longitude=-87.6309,
        destination_label="54th/Cermak", destination_latitude=41.8518, destination_longitude=-87.7567,
        boarding_int_id=40380, boarding_text_id=None,
        alighting_int_id=40580, alighting_text_id=None,
        schedule_headway_seconds=900.0, cadence_seconds=300.0, priority=4,
    ),

    # --- CTA Bus 22 (Clark): Belmont <-> Adams ---
    Corridor(
        corridor_id="cta-bus-22-belmont-adams-sb",
        mode="bus", line="22", direction="southbound",
        origin_label="Clark & Belmont", origin_latitude=41.9401, origin_longitude=-87.6509,
        destination_label="Clark & Adams", destination_latitude=41.8791, destination_longitude=-87.6309,
        boarding_int_id=1828, boarding_text_id=None,
        alighting_int_id=1869, alighting_text_id=None,
        schedule_headway_seconds=600.0, cadence_seconds=300.0, priority=5,
    ),
    Corridor(
        corridor_id="cta-bus-22-belmont-adams-nb",
        mode="bus", line="22", direction="northbound",
        origin_label="Dearborn & Grand", origin_latitude=41.8918, origin_longitude=-87.6296,
        destination_label="Clark & Belmont", destination_latitude=41.9399, destination_longitude=-87.6505,
        boarding_int_id=14767, boarding_text_id=None,
        alighting_int_id=1921, alighting_text_id=None,
        schedule_headway_seconds=600.0, cadence_seconds=300.0, priority=5,
    ),

    # --- CTA Brown Line: Kimball <-> Merchandise Mart ---
    Corridor(
        corridor_id="cta-brown-kimball-mart-sb",
        mode="L", line="Brn", direction="southbound",
        origin_label="Kimball", origin_latitude=41.9679, origin_longitude=-87.7131,
        destination_label="Merchandise Mart", destination_latitude=41.8890, destination_longitude=-87.6339,
        boarding_int_id=41290, boarding_text_id=None,
        alighting_int_id=40460, alighting_text_id=None,
        schedule_headway_seconds=600.0, cadence_seconds=300.0, priority=2,
    ),
    Corridor(
        corridor_id="cta-brown-kimball-mart-nb",
        mode="L", line="Brn", direction="northbound",
        origin_label="Merchandise Mart", origin_latitude=41.8890, origin_longitude=-87.6339,
        destination_label="Kimball", destination_latitude=41.9679, destination_longitude=-87.7131,
        boarding_int_id=40460, boarding_text_id=None,
        alighting_int_id=41290, alighting_text_id=None,
        schedule_headway_seconds=600.0, cadence_seconds=300.0, priority=2,
    ),

    # --- CTA Green Line - West: Harlem/Lake <-> Clark/Lake ---
    Corridor(
        corridor_id="cta-green-harlem-loop-eb",
        mode="L", line="G", direction="eastbound",
        origin_label="Harlem/Lake", origin_latitude=41.8868, origin_longitude=-87.8032,
        destination_label="Clark/Lake", destination_latitude=41.8857, destination_longitude=-87.6309,
        boarding_int_id=40020, boarding_text_id=None,
        alighting_int_id=40380, alighting_text_id=None,
        schedule_headway_seconds=600.0, cadence_seconds=300.0, priority=3,
    ),
    Corridor(
        corridor_id="cta-green-harlem-loop-wb",
        mode="L", line="G", direction="westbound",
        origin_label="Clark/Lake", origin_latitude=41.8857, origin_longitude=-87.6309,
        destination_label="Harlem/Lake", destination_latitude=41.8868, destination_longitude=-87.8032,
        boarding_int_id=40380, boarding_text_id=None,
        alighting_int_id=40020, alighting_text_id=None,
        schedule_headway_seconds=600.0, cadence_seconds=300.0, priority=3,
    ),

    # --- CTA Green Line - South: Cottage Grove <-> Roosevelt ---
    Corridor(
        corridor_id="cta-green-cottage-roosevelt-nb",
        mode="L", line="G", direction="northbound",
        origin_label="Cottage Grove", origin_latitude=41.7803, origin_longitude=-87.6059,
        destination_label="Roosevelt", destination_latitude=41.8674, destination_longitude=-87.6274,
        boarding_int_id=40720, boarding_text_id=None,
        alighting_int_id=41400, alighting_text_id=None,
        schedule_headway_seconds=900.0, cadence_seconds=300.0, priority=4,
    ),
    Corridor(
        corridor_id="cta-green-cottage-roosevelt-sb",
        mode="L", line="G", direction="southbound",
        origin_label="Roosevelt", origin_latitude=41.8674, origin_longitude=-87.6274,
        destination_label="Cottage Grove", destination_latitude=41.7803, destination_longitude=-87.6059,
        boarding_int_id=41400, boarding_text_id=None,
        alighting_int_id=40720, alighting_text_id=None,
        schedule_headway_seconds=900.0, cadence_seconds=300.0, priority=4,
    ),

    # --- CTA Orange Line: Midway <-> Clark/Lake ---
    Corridor(
        corridor_id="cta-orange-midway-loop-nb",
        mode="L", line="Org", direction="northbound",
        origin_label="Midway", origin_latitude=41.7866, origin_longitude=-87.7379,
        destination_label="Clark/Lake", destination_latitude=41.8857, destination_longitude=-87.6309,
        boarding_int_id=40930, boarding_text_id=None,
        alighting_int_id=40380, alighting_text_id=None,
        schedule_headway_seconds=600.0, cadence_seconds=300.0, priority=3,
    ),
    Corridor(
        corridor_id="cta-orange-midway-loop-sb",
        mode="L", line="Org", direction="southbound",
        origin_label="Clark/Lake", origin_latitude=41.8857, origin_longitude=-87.6309,
        destination_label="Midway", destination_latitude=41.7866, destination_longitude=-87.7379,
        boarding_int_id=40380, boarding_text_id=None,
        alighting_int_id=40930, alighting_text_id=None,
        schedule_headway_seconds=600.0, cadence_seconds=300.0, priority=3,
    ),

    # --- CTA Purple Line: Linden <-> Howard ---
    Corridor(
        corridor_id="cta-purple-linden-howard-sb",
        mode="L", line="P", direction="southbound",
        origin_label="Linden", origin_latitude=42.0732, origin_longitude=-87.6907,
        destination_label="Howard", destination_latitude=42.0191, destination_longitude=-87.6729,
        boarding_int_id=41050, boarding_text_id=None,
        alighting_int_id=40900, alighting_text_id=None,
        schedule_headway_seconds=720.0, cadence_seconds=300.0, priority=3,
    ),
    Corridor(
        corridor_id="cta-purple-linden-howard-nb",
        mode="L", line="P", direction="northbound",
        origin_label="Howard", origin_latitude=42.0191, origin_longitude=-87.6729,
        destination_label="Linden", destination_latitude=42.0732, destination_longitude=-87.6907,
        boarding_int_id=40900, boarding_text_id=None,
        alighting_int_id=41050, alighting_text_id=None,
        schedule_headway_seconds=720.0, cadence_seconds=300.0, priority=3,
    ),

    # --- CTA Yellow Line: Dempster-Skokie <-> Howard ---
    Corridor(
        corridor_id="cta-yellow-skokie-howard-sb",
        mode="L", line="Y", direction="southbound",
        origin_label="Dempster-Skokie", origin_latitude=42.0390, origin_longitude=-87.7519,
        destination_label="Howard", destination_latitude=42.0191, destination_longitude=-87.6729,
        boarding_int_id=40140, boarding_text_id=None,
        alighting_int_id=40900, alighting_text_id=None,
        schedule_headway_seconds=600.0, cadence_seconds=300.0, priority=4,
    ),
    Corridor(
        corridor_id="cta-yellow-skokie-howard-nb",
        mode="L", line="Y", direction="northbound",
        origin_label="Howard", origin_latitude=42.0191, origin_longitude=-87.6729,
        destination_label="Dempster-Skokie", destination_latitude=42.0390, destination_longitude=-87.7519,
        boarding_int_id=40900, boarding_text_id=None,
        alighting_int_id=40140, alighting_text_id=None,
        schedule_headway_seconds=600.0, cadence_seconds=300.0, priority=4,
    ),

    # --- CTA Blue Line - Forest Park: Forest Park <-> Clark/Lake ---
    Corridor(
        corridor_id="cta-blue-fp-loop-eb",
        mode="L", line="Blue", direction="eastbound",
        origin_label="Forest Park", origin_latitude=41.8743, origin_longitude=-87.8173,
        destination_label="Clark/Lake", destination_latitude=41.8857, destination_longitude=-87.6309,
        boarding_int_id=40390, boarding_text_id=None,
        alighting_int_id=40380, alighting_text_id=None,
        schedule_headway_seconds=600.0, cadence_seconds=300.0, priority=3,
    ),
    Corridor(
        corridor_id="cta-blue-fp-loop-wb",
        mode="L", line="Blue", direction="westbound",
        origin_label="Clark/Lake", origin_latitude=41.8857, origin_longitude=-87.6309,
        destination_label="Forest Park", destination_latitude=41.8743, destination_longitude=-87.8173,
        boarding_int_id=40380, boarding_text_id=None,
        alighting_int_id=40390, alighting_text_id=None,
        schedule_headway_seconds=600.0, cadence_seconds=300.0, priority=3,
    ),

    # --- Metra BNSF: Aurora <-> Union Station ---
    Corridor(
        corridor_id="metra-bnsf-aurora-cus-ib",
        mode="metra", line="BNSF", direction="inbound",
        origin_label="Aurora", origin_latitude=41.7608, origin_longitude=-88.3083,
        destination_label="Chicago Union Station", destination_latitude=41.8789, destination_longitude=-87.6389,
        boarding_int_id=0, boarding_text_id="AURORA",
        alighting_int_id=0, alighting_text_id="CUS",
        schedule_headway_seconds=1800.0, cadence_seconds=300.0, priority=2,
    ),
    Corridor(
        corridor_id="metra-bnsf-aurora-cus-ob",
        mode="metra", line="BNSF", direction="outbound",
        origin_label="Chicago Union Station", origin_latitude=41.8789, origin_longitude=-87.6389,
        destination_label="Aurora", destination_latitude=41.7608, destination_longitude=-88.3083,
        boarding_int_id=0, boarding_text_id="CUS",
        alighting_int_id=0, alighting_text_id="AURORA",
        schedule_headway_seconds=1800.0, cadence_seconds=300.0, priority=2,
    ),

    # --- Metra Electric (MED): 93rd/South Chicago <-> Millennium ---
    Corridor(
        corridor_id="metra-me-93rd-millennium-ib",
        mode="metra", line="ME", direction="inbound",
        origin_label="South Chicago (93rd)", origin_latitude=41.7267, origin_longitude=-87.5478,
        destination_label="Millennium Station", destination_latitude=41.8842, destination_longitude=-87.6231,
        boarding_int_id=0, boarding_text_id="93RD-SC",
        alighting_int_id=0, alighting_text_id="MILLENNIUM",
        schedule_headway_seconds=1800.0, cadence_seconds=300.0, priority=3,
    ),
    Corridor(
        corridor_id="metra-me-93rd-millennium-ob",
        mode="metra", line="ME", direction="outbound",
        origin_label="Millennium Station", origin_latitude=41.8842, origin_longitude=-87.6231,
        destination_label="South Chicago (93rd)", destination_latitude=41.7267, destination_longitude=-87.5478,
        boarding_int_id=0, boarding_text_id="MILLENNIUM",
        alighting_int_id=0, alighting_text_id="93RD-SC",
        schedule_headway_seconds=1800.0, cadence_seconds=300.0, priority=3,
    ),

    # --- CTA Bus 66 (Chicago Ave): Michigan <-> Western ---
    Corridor(
        corridor_id="cta-bus-66-michigan-western-wb",
        mode="bus", line="66", direction="westbound",
        origin_label="Chicago & Michigan", origin_latitude=41.8968, origin_longitude=-87.6245,
        destination_label="Chicago & Western", destination_latitude=41.8958, destination_longitude=-87.6873,
        boarding_int_id=599, boarding_text_id=None,
        alighting_int_id=15203, alighting_text_id=None,
        schedule_headway_seconds=480.0, cadence_seconds=300.0, priority=3,
    ),
    Corridor(
        corridor_id="cta-bus-66-michigan-western-eb",
        mode="bus", line="66", direction="eastbound",
        origin_label="Chicago & Western", origin_latitude=41.8957, origin_longitude=-87.6870,
        destination_label="Chicago & Michigan", destination_latitude=41.8967, destination_longitude=-87.6245,
        boarding_int_id=548, boarding_text_id=None,
        alighting_int_id=580, alighting_text_id=None,
        schedule_headway_seconds=480.0, cadence_seconds=300.0, priority=3,
    ),

    # --- CTA Bus 9 (Ashland): Belmont <-> 87th ---
    Corridor(
        corridor_id="cta-bus-9-belmont-87th-sb",
        mode="bus", line="9", direction="southbound",
        origin_label="Ashland & Belmont", origin_latitude=41.9400, origin_longitude=-87.6688,
        destination_label="Ashland & 87th", destination_latitude=41.7355, destination_longitude=-87.6632,
        boarding_int_id=6003, boarding_text_id=None,
        alighting_int_id=15249, alighting_text_id=None,
        schedule_headway_seconds=480.0, cadence_seconds=300.0, priority=3,
    ),
    Corridor(
        corridor_id="cta-bus-9-belmont-87th-nb",
        mode="bus", line="9", direction="northbound",
        origin_label="Ashland & 87th", origin_latitude=41.7356, origin_longitude=-87.6629,
        destination_label="Ashland & Belmont", destination_latitude=41.9395, destination_longitude=-87.6685,
        boarding_int_id=6155, boarding_text_id=None,
        alighting_int_id=6272, alighting_text_id=None,
        schedule_headway_seconds=480.0, cadence_seconds=300.0, priority=3,
    ),

    # --- CTA Bus 79 (79th St): Halsted <-> Stony Island ---
    Corridor(
        corridor_id="cta-bus-79-halsted-stony-eb",
        mode="bus", line="79", direction="eastbound",
        origin_label="79th & Halsted", origin_latitude=41.7506, origin_longitude=-87.6442,
        destination_label="79th & Stony Island", destination_latitude=41.7514, destination_longitude=-87.5849,
        boarding_int_id=2762, boarding_text_id=None,
        alighting_int_id=2795, alighting_text_id=None,
        schedule_headway_seconds=420.0, cadence_seconds=300.0, priority=4,
    ),
    Corridor(
        corridor_id="cta-bus-79-halsted-stony-wb",
        mode="bus", line="79", direction="westbound",
        origin_label="79th & Stony Island", origin_latitude=41.7516, origin_longitude=-87.5864,
        destination_label="79th & Halsted", destination_latitude=41.7507, destination_longitude=-87.6443,
        boarding_int_id=2621, boarding_text_id=None,
        alighting_int_id=17349, alighting_text_id=None,
        schedule_headway_seconds=420.0, cadence_seconds=300.0, priority=4,
    ),

    # === INTERMEDIATE CORRIDORS ===
    # Mid-line OD pairs picked for transfer relevance + station ridership.
    # Lower priority than the endpoint corridors above so endpoints are
    # always predicted first when poll budget is tight.

    # Red Line intermediates
    Corridor(
        corridor_id="cta-red-fullerton-roosevelt-sb",
        mode="L", line="Red", direction="southbound",
        origin_label="Fullerton", origin_latitude=41.9251, origin_longitude=-87.6529,
        destination_label="Roosevelt", destination_latitude=41.8674, destination_longitude=-87.6274,
        boarding_int_id=41220, boarding_text_id=None,
        alighting_int_id=41400, alighting_text_id=None,
        schedule_headway_seconds=540.0, cadence_seconds=300.0, priority=5,
    ),
    Corridor(
        corridor_id="cta-red-fullerton-roosevelt-nb",
        mode="L", line="Red", direction="northbound",
        origin_label="Roosevelt", origin_latitude=41.8674, origin_longitude=-87.6274,
        destination_label="Fullerton", destination_latitude=41.9251, destination_longitude=-87.6529,
        boarding_int_id=41400, boarding_text_id=None,
        alighting_int_id=41220, alighting_text_id=None,
        schedule_headway_seconds=540.0, cadence_seconds=300.0, priority=5,
    ),
    Corridor(
        corridor_id="cta-red-wilson-chicago-sb",
        mode="L", line="Red", direction="southbound",
        origin_label="Wilson", origin_latitude=41.9643, origin_longitude=-87.6576,
        destination_label="Chicago", destination_latitude=41.8967, destination_longitude=-87.6282,
        boarding_int_id=40540, boarding_text_id=None,
        alighting_int_id=41450, alighting_text_id=None,
        schedule_headway_seconds=540.0, cadence_seconds=300.0, priority=5,
    ),
    Corridor(
        corridor_id="cta-red-wilson-chicago-nb",
        mode="L", line="Red", direction="northbound",
        origin_label="Chicago", origin_latitude=41.8967, origin_longitude=-87.6282,
        destination_label="Wilson", destination_latitude=41.9643, destination_longitude=-87.6576,
        boarding_int_id=41450, boarding_text_id=None,
        alighting_int_id=40540, alighting_text_id=None,
        schedule_headway_seconds=540.0, cadence_seconds=300.0, priority=5,
    ),
    Corridor(
        corridor_id="cta-red-roosevelt-79th-sb",
        mode="L", line="Red", direction="southbound",
        origin_label="Roosevelt", origin_latitude=41.8674, origin_longitude=-87.6274,
        destination_label="79th", destination_latitude=41.7504, destination_longitude=-87.6251,
        boarding_int_id=41400, boarding_text_id=None,
        alighting_int_id=40240, alighting_text_id=None,
        schedule_headway_seconds=540.0, cadence_seconds=300.0, priority=5,
    ),
    Corridor(
        corridor_id="cta-red-roosevelt-79th-nb",
        mode="L", line="Red", direction="northbound",
        origin_label="79th", origin_latitude=41.7504, origin_longitude=-87.6251,
        destination_label="Roosevelt", destination_latitude=41.8674, destination_longitude=-87.6274,
        boarding_int_id=40240, boarding_text_id=None,
        alighting_int_id=41400, alighting_text_id=None,
        schedule_headway_seconds=540.0, cadence_seconds=300.0, priority=5,
    ),

    # Blue Line intermediates
    Corridor(
        corridor_id="cta-blue-jefferson-loop-eb",
        mode="L", line="Blue", direction="eastbound",
        origin_label="Jefferson Park", origin_latitude=41.9706, origin_longitude=-87.7609,
        destination_label="Clark/Lake", destination_latitude=41.8857, destination_longitude=-87.6309,
        boarding_int_id=41280, boarding_text_id=None,
        alighting_int_id=40380, alighting_text_id=None,
        schedule_headway_seconds=600.0, cadence_seconds=300.0, priority=5,
    ),
    Corridor(
        corridor_id="cta-blue-jefferson-loop-wb",
        mode="L", line="Blue", direction="westbound",
        origin_label="Clark/Lake", origin_latitude=41.8857, origin_longitude=-87.6309,
        destination_label="Jefferson Park", destination_latitude=41.9706, destination_longitude=-87.7609,
        boarding_int_id=40380, boarding_text_id=None,
        alighting_int_id=41280, alighting_text_id=None,
        schedule_headway_seconds=600.0, cadence_seconds=300.0, priority=5,
    ),
    Corridor(
        corridor_id="cta-blue-logan-loop-eb",
        mode="L", line="Blue", direction="eastbound",
        origin_label="Logan Square", origin_latitude=41.9297, origin_longitude=-87.7085,
        destination_label="Clark/Lake", destination_latitude=41.8857, destination_longitude=-87.6309,
        boarding_int_id=41020, boarding_text_id=None,
        alighting_int_id=40380, alighting_text_id=None,
        schedule_headway_seconds=600.0, cadence_seconds=300.0, priority=5,
    ),
    Corridor(
        corridor_id="cta-blue-logan-loop-wb",
        mode="L", line="Blue", direction="westbound",
        origin_label="Clark/Lake", origin_latitude=41.8857, origin_longitude=-87.6309,
        destination_label="Logan Square", destination_latitude=41.9297, destination_longitude=-87.7085,
        boarding_int_id=40380, boarding_text_id=None,
        alighting_int_id=41020, alighting_text_id=None,
        schedule_headway_seconds=600.0, cadence_seconds=300.0, priority=5,
    ),

    # Brown Line intermediate
    Corridor(
        corridor_id="cta-brown-western-mart-sb",
        mode="L", line="Brn", direction="southbound",
        origin_label="Western (Brown)", origin_latitude=41.9662, origin_longitude=-87.6885,
        destination_label="Merchandise Mart", destination_latitude=41.8890, destination_longitude=-87.6339,
        boarding_int_id=41480, boarding_text_id=None,
        alighting_int_id=40460, alighting_text_id=None,
        schedule_headway_seconds=600.0, cadence_seconds=300.0, priority=5,
    ),
    Corridor(
        corridor_id="cta-brown-western-mart-nb",
        mode="L", line="Brn", direction="northbound",
        origin_label="Merchandise Mart", origin_latitude=41.8890, origin_longitude=-87.6339,
        destination_label="Western (Brown)", destination_latitude=41.9662, destination_longitude=-87.6885,
        boarding_int_id=40460, boarding_text_id=None,
        alighting_int_id=41480, alighting_text_id=None,
        schedule_headway_seconds=600.0, cadence_seconds=300.0, priority=5,
    ),

    # Green Line intermediates
    Corridor(
        corridor_id="cta-green-garfield-roosevelt-nb",
        mode="L", line="G", direction="northbound",
        origin_label="Garfield (Green)", origin_latitude=41.7952, origin_longitude=-87.6183,
        destination_label="Roosevelt", destination_latitude=41.8674, destination_longitude=-87.6274,
        boarding_int_id=40510, boarding_text_id=None,
        alighting_int_id=41400, alighting_text_id=None,
        schedule_headway_seconds=900.0, cadence_seconds=300.0, priority=5,
    ),
    Corridor(
        corridor_id="cta-green-garfield-roosevelt-sb",
        mode="L", line="G", direction="southbound",
        origin_label="Roosevelt", origin_latitude=41.8674, origin_longitude=-87.6274,
        destination_label="Garfield (Green)", destination_latitude=41.7952, destination_longitude=-87.6183,
        boarding_int_id=41400, boarding_text_id=None,
        alighting_int_id=40510, alighting_text_id=None,
        schedule_headway_seconds=900.0, cadence_seconds=300.0, priority=5,
    ),
    Corridor(
        corridor_id="cta-green-ashland-loop-eb",
        mode="L", line="G", direction="eastbound",
        origin_label="Ashland (Green)", origin_latitude=41.8853, origin_longitude=-87.6670,
        destination_label="Clark/Lake", destination_latitude=41.8857, destination_longitude=-87.6309,
        boarding_int_id=40170, boarding_text_id=None,
        alighting_int_id=40380, alighting_text_id=None,
        schedule_headway_seconds=600.0, cadence_seconds=300.0, priority=5,
    ),
    Corridor(
        corridor_id="cta-green-ashland-loop-wb",
        mode="L", line="G", direction="westbound",
        origin_label="Clark/Lake", origin_latitude=41.8857, origin_longitude=-87.6309,
        destination_label="Ashland (Green)", destination_latitude=41.8853, destination_longitude=-87.6670,
        boarding_int_id=40380, boarding_text_id=None,
        alighting_int_id=40170, alighting_text_id=None,
        schedule_headway_seconds=600.0, cadence_seconds=300.0, priority=5,
    ),

    # Orange Line intermediate
    Corridor(
        corridor_id="cta-orange-pulaski-loop-nb",
        mode="L", line="Org", direction="northbound",
        origin_label="Pulaski (Orange)", origin_latitude=41.7998, origin_longitude=-87.7245,
        destination_label="Clark/Lake", destination_latitude=41.8857, destination_longitude=-87.6309,
        boarding_int_id=40960, boarding_text_id=None,
        alighting_int_id=40380, alighting_text_id=None,
        schedule_headway_seconds=600.0, cadence_seconds=300.0, priority=5,
    ),
    Corridor(
        corridor_id="cta-orange-pulaski-loop-sb",
        mode="L", line="Org", direction="southbound",
        origin_label="Clark/Lake", origin_latitude=41.8857, origin_longitude=-87.6309,
        destination_label="Pulaski (Orange)", destination_latitude=41.7998, destination_longitude=-87.7245,
        boarding_int_id=40380, boarding_text_id=None,
        alighting_int_id=40960, alighting_text_id=None,
        schedule_headway_seconds=600.0, cadence_seconds=300.0, priority=5,
    ),

    # Pink Line intermediate
    Corridor(
        corridor_id="cta-pink-polk-loop-eb",
        mode="L", line="Pink", direction="eastbound",
        origin_label="Polk", origin_latitude=41.8716, origin_longitude=-87.6695,
        destination_label="Clark/Lake", destination_latitude=41.8857, destination_longitude=-87.6309,
        boarding_int_id=41030, boarding_text_id=None,
        alighting_int_id=40380, alighting_text_id=None,
        schedule_headway_seconds=900.0, cadence_seconds=300.0, priority=5,
    ),
    Corridor(
        corridor_id="cta-pink-polk-loop-wb",
        mode="L", line="Pink", direction="westbound",
        origin_label="Clark/Lake", origin_latitude=41.8857, origin_longitude=-87.6309,
        destination_label="Polk", destination_latitude=41.8716, destination_longitude=-87.6695,
        boarding_int_id=40380, boarding_text_id=None,
        alighting_int_id=41030, alighting_text_id=None,
        schedule_headway_seconds=900.0, cadence_seconds=300.0, priority=5,
    ),
)


def by_id() -> dict[str, Corridor]:
    return {c.corridor_id: c for c in SEED_CORRIDORS}


def by_mode(mode: str) -> tuple[Corridor, ...]:
    return tuple(c for c in SEED_CORRIDORS if c.mode == mode)


def seed_corridors(conn: duckdb.DuckDBPyConnection, *, now: datetime) -> int:
    """Insert any missing corridors into the ``corridors`` table.

    Upserts on ``corridor_id``: re-seeding is safe and refreshes metadata
    on any corridor row whose seed definition changed (e.g. corrected
    coordinates, adjusted cadence).
    """
    rows = [
        (
            c.corridor_id, c.mode, c.line, c.direction,
            c.origin_label, c.origin_latitude, c.origin_longitude,
            c.destination_label, c.destination_latitude, c.destination_longitude,
            c.boarding_int_id, c.boarding_text_id,
            c.alighting_int_id, c.alighting_text_id,
            c.schedule_headway_seconds, c.cadence_seconds, c.priority,
            True, now,
        )
        for c in SEED_CORRIDORS
    ]
    conn.executemany(
        """
        INSERT INTO corridors (
            corridor_id, mode, line, direction,
            origin_label, origin_latitude, origin_longitude,
            destination_label, destination_latitude, destination_longitude,
            boarding_int_id, boarding_text_id,
            alighting_int_id, alighting_text_id,
            schedule_headway_seconds, cadence_seconds, priority,
            is_active, seeded_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (corridor_id) DO UPDATE SET
            mode = excluded.mode,
            line = excluded.line,
            direction = excluded.direction,
            origin_label = excluded.origin_label,
            origin_latitude = excluded.origin_latitude,
            origin_longitude = excluded.origin_longitude,
            destination_label = excluded.destination_label,
            destination_latitude = excluded.destination_latitude,
            destination_longitude = excluded.destination_longitude,
            boarding_int_id = excluded.boarding_int_id,
            boarding_text_id = excluded.boarding_text_id,
            alighting_int_id = excluded.alighting_int_id,
            alighting_text_id = excluded.alighting_text_id,
            schedule_headway_seconds = excluded.schedule_headway_seconds,
            cadence_seconds = excluded.cadence_seconds,
            priority = excluded.priority,
            is_active = excluded.is_active
        """,
        rows,
    )
    return len(rows)


def due_corridors(
    conn: duckdb.DuckDBPyConnection,
    *,
    now: datetime,
    enabled_modes: Iterable[str],
) -> list[Corridor]:
    """Return active corridors whose cadence window has elapsed since their
    last prediction.

    A corridor is due if ``last_predicted_at`` is NULL, or if
    ``(now - last_predicted_at) >= cadence_seconds``. Ordered by priority
    (lowest int first) then by how long they've been waiting.
    """
    enabled = tuple(enabled_modes)
    if not enabled:
        return []
    placeholders = ",".join(["?"] * len(enabled))
    rows = conn.execute(
        f"""
        SELECT corridor_id, mode, line, direction,
               origin_label, origin_latitude, origin_longitude,
               destination_label, destination_latitude, destination_longitude,
               boarding_int_id, boarding_text_id,
               alighting_int_id, alighting_text_id,
               schedule_headway_seconds, cadence_seconds, priority,
               last_predicted_at
          FROM corridors
         WHERE is_active = TRUE
           AND mode IN ({placeholders})
           AND (
                last_predicted_at IS NULL
                OR EPOCH(? - last_predicted_at) >= cadence_seconds
           )
         ORDER BY priority ASC,
                  COALESCE(last_predicted_at, TIMESTAMPTZ '1970-01-01') ASC
        """,
        list(enabled) + [now],
    ).fetchall()
    out: list[Corridor] = []
    for r in rows:
        out.append(
            Corridor(
                corridor_id=r[0], mode=r[1], line=r[2], direction=r[3],
                origin_label=r[4], origin_latitude=r[5], origin_longitude=r[6],
                destination_label=r[7], destination_latitude=r[8], destination_longitude=r[9],
                boarding_int_id=r[10], boarding_text_id=r[11],
                alighting_int_id=r[12], alighting_text_id=r[13],
                schedule_headway_seconds=r[14], cadence_seconds=r[15], priority=r[16],
            )
        )
    return out


def mark_predicted(
    conn: duckdb.DuckDBPyConnection,
    *,
    corridor_id: str,
    at: datetime,
) -> None:
    conn.execute(
        "UPDATE corridors SET last_predicted_at = ? WHERE corridor_id = ?",
        [at, corridor_id],
    )
