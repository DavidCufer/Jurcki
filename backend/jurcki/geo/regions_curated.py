"""Hand-curated mushroom regions that GeoNames represents poorly or not at all.

Each region is an approximate ellipse: centre (lat, lon), semi-major ``a_km``,
semi-minor ``b_km`` and azimuth ``az`` of the major axis (degrees clockwise
from north).  There is no authoritative polygon available offline (the
geoBoundaries LFS files are on a host that is blocked here), so the extents
below are *approximate, hand-estimated* from the author's knowledge of the
geography, with centres cross-checked against GeoNames features where one
exists (``tests`` and ``check_against_geonames`` print the offsets).  They are
meant for "which part of Slovenia is this post about", not for cadastral use.

``kind`` is one of: mountain, plateau, forest, valley, region (statistical /
historical region), area (small named area).
``aliases`` are nominative alternative names; ``(name, weight)`` tuples lower
the confidence of looser aliases (e.g. adjective forms "pohorski").
``strict`` means the name is also an ordinary word, so a match needs a
capitalised token and a preposition in front of it.
"""
from __future__ import annotations

# name, lat, lon, a_km, b_km, az, kind, aliases, basis, strict
_R = []


def R(name, lat, lon, a, b, az, kind, aliases=(), basis="", strict=False, weight=1.0):
    _R.append(dict(name=name, lat=lat, lon=lon, a_km=a, b_km=b, az=az, kind=kind,
                   aliases=list(aliases), basis=basis, strict=strict, weight=weight))


GN_MTS = "GeoNames MTS point + hand-estimated extent"
HAND = "hand-estimated centre and extent (approximate)"
STAT = "approximate extent of the traditional region (hand-estimated; not an official polygon)"

# ---- mountains / plateaus / forests ----------------------------------------
R("Pohorje", 46.47, 15.42, 22, 9, 80, "mountain",
  [("Pohorski", .8), ("Pohorska", .8)], "GeoNames MTS 1381 (46.53,15.47) re-centred; ridge Ribnica na Pohorju - Maribor ~45x18 km")
R("Mariborsko Pohorje", 46.523, 15.585, 5, 4, 80, "mountain", [("Mariborska koča", 1)], "ski resort area above Maribor (hand-estimated)")
R("Rogla", 46.456, 15.332, 5, 4, 80, "mountain", [], "GeoNames MT 1097 (46.453,15.332) + surrounding plateau forests")
R("Areh", 46.505, 15.535, 3.5, 3, 80, "mountain", [], "hand-estimated: Areh plateau/ski area, E Pohorje")
R("Kozjak", 46.60, 15.40, 16, 5, 105, "mountain", [], "GeoNames MT 3807 (46.596,15.380); ridge N of Drava, Maribor-Radlje (hand extent)")
R("Paški Kozjak", 46.373, 15.215, 5, 4, 0, "mountain", [], "GeoNames MTS 2193 (46.373,15.215)")
R("Kočevski Rog", 45.65, 15.00, 12, 7, 135, "forest", [("Rog", .75)],
  "GeoNames MT 2267 (45.649,15.030); large forest plateau between Kočevje, Črnomelj and Dolenjske Toplice")
R("Kočevsko", 45.62, 14.85, 22, 16, 120, "region", [("Kočevski", .7)], "region around Kočevje (hand-estimated)")
R("Snežnik", 45.589, 14.453, 9, 6, 135, "mountain", [("Snežniška planota", 1), ("Snežniški", .75)],
  "GeoNames MTS 846 (45.589,14.453); massif plus forest plateau (Sviščaki, Leskova dolina)")
R("Javorniki", 45.73, 14.28, 22, 8, 140, "forest", [("Javorniški", .7)], "GeoNames MTS 2403 (45.733,14.283); NW-SE forested ridge Postojna-Snežnik")
R("Trnovski gozd", 45.96, 13.90, 18, 9, 135, "forest", [("Trnovski", .8)],
  "GeoNames FRST 496 (45.961,13.897); karst forest plateau between Soča/Idrijca and Vipava valley")
R("Nanos", 45.772, 14.053, 9, 3.5, 75, "mountain", [], "GeoNames MTS 1686 (45.772,14.053); elongated plateau ridge")
R("Hrušica", 45.849, 14.177, 10, 5, 110, "forest", [], "GeoNames MTS 2483 (45.849,14.177). AMBIGUOUS: also a village near Jesenice and others", weight=.85)
R("Banjška planota", 46.07, 13.73, 9, 5, 0, "plateau", ["Banjšice", "Banjška"], "GeoNames PLAT 3330 (46.094,13.726) + PPL Banjšice (46.05,13.717)")
R("Pokljuka", 46.35, 13.95, 10, 6, 100, "plateau", [], "GeoNames RGN 1380 (46.367,13.933); forest plateau above Bled/Bohinj")
R("Jelovica", 46.28, 14.08, 11, 8, 100, "plateau", [], "hand-estimated; GeoNames MT 2390 (46.30,14.133) is only a peak")
R("Mežakla", 46.43, 14.00, 6, 3.5, 90, "plateau", [], "GeoNames RDGE 1776 (46.433,14.0) + extent")
R("Menina planina", 46.25, 14.82, 12, 5, 80, "plateau", [("Menina", .9)], "GeoNames MTS 1790 (46.25,14.817)")
R("Gorjanci", 45.77, 15.32, 15, 5, 110, "mountain", [("Žumberak", .8), ("Žumberačka Gora", .8)],
  "GeoNames MTS 11 'Žumberačka Gora' (45.75,15.333); ridge Novo mesto - Metlika on the SI/HR border")
R("Bohor", 46.081, 15.457, 9, 3.5, 80, "mountain", [], "GeoNames MT 3236 (46.081,15.457); ridge")
R("Haloze", 46.34, 15.95, 15, 7, 100, "area", [("Haloški", .8), ("Haloški gozd", 1)], "hills S of Ptuj along the Croatian border (hand-estimated)")
R("Slovenske gorice", 46.60, 15.90, 22, 10, 85, "area", [], "GeoNames HLLS 895 (46.583,15.917); Maribor - Ormož hill country")
R("Goričko", 46.80, 16.10, 16, 9, 80, "area", [], "NE hills around Grad, Cankova, Kuzma (hand-estimated)")
R("Kozjansko", 46.06, 15.60, 15, 11, 60, "region", [("Kozjanski", .8)], "area Kozje-Podsreda-Podčetrtek-Bistrica ob Sotli (hand-estimated)")
R("Posavsko hribovje", 46.10, 14.98, 22, 9, 85, "mountain", [], "hills between Ljubljana basin and Sevnica incl. Kum, Mrzlica (hand-estimated)")
R("Krimsko hribovje", 45.93, 14.50, 8, 5, 100, "mountain", [("Krim", 1), ("Mokrc", .8), ("Krimski", .7)],
  "GeoNames MT 2164 Krim (45.929,14.471); Mokrc position uncertain (hand-estimated)")
R("Bloke", 45.78, 14.50, 9, 6, 90, "plateau", [("Bloška planota", 1), ("Bloški", .7)], "GeoNames PPL Velike Bloke (45.784,14.481) + plateau", strict=True)
R("Ribniško", 45.74, 14.73, 8, 5, 90, "area", [("Ribniška dolina", 1), ("Ribniški", .7)], "area around Ribnica (hand-estimated)")
R("Suha krajina", 45.82, 14.95, 10, 5, 100, "area", [], "area around Žužemberk-Dvor (hand-estimated)")

# ---- regions -----------------------------------------------------------------
R("Notranjska", 45.74, 14.33, 28, 17, 135, "region", [("Notranjsko", 1)],
  "GeoNames PLN 'Notranjsko' (45.767,14.433) + traditional extent Logatec-Postojna-Ilirska Bistrica-Loški Potok")
R("Dolenjska", 45.85, 15.00, 40, 20, 100, "region", [("Dolenjsko", 1)], "GeoNames RGN 2882 (45.833,15.167) 'Lower Carniola', re-centred; " + STAT)
R("Gorenjska", 46.32, 14.20, 38, 24, 110, "region", [("Gorenjsko", 1)], "GeoNames RGN 2647 (46.333,14.167); " + STAT)
R("Štajerska", 46.30, 15.60, 50, 28, 85, "region", [("Štajersko", 1)], STAT)
R("Koroška", 46.50, 14.98, 22, 11, 95, "region", [("Koroško", 1)], "GeoNames RGN 7707 'Carinthia' (46.53,14.95); Slovenian Carinthia only")
R("Primorska", 45.95, 13.85, 45, 25, 20, "region", [("Primorsko", 1)], "Slovene Littoral (hand-estimated; GeoNames only has an English-named point)")
R("Bela krajina", 45.59, 15.22, 17, 9, 100, "region", [("Belokranjsko", 1), ("Belokranjski", .8)], "GeoNames RGN 3313 'White Carniola' (45.583,15.25)")
R("Prekmurje", 46.65, 16.25, 26, 17, 60, "region", [("Pomurje", .9), ("Prekmurski", .8)], "GeoNames AREA 3982 (46.719,16.272) / RGN 1297; Mura-Rába region")
R("Kras", 45.78, 13.85, 22, 7, 135, "region", [("Kraško", .9), ("Kraški", .8)], "GeoNames RGN 2176 'Karst region' (45.806,13.749); Sežana-Komen-Divača", strict=True)
R("Brkini", 45.58, 14.105, 14, 6, 130, "area", [("Brkinski", .8)], "GeoNames AREA 3132 (45.580,14.105)")
R("Idrijsko", 46.02, 14.00, 15, 11, 90, "region", [("Idrijski", .75)], "area around Idrija (hand-estimated)")
R("Cerkljansko", 46.13, 13.95, 10, 8, 90, "region", [("Cerkljanski", .75)], "area around Cerkno (hand-estimated)")
R("Škofjeloško hribovje", 46.15, 14.17, 14, 8, 100, "mountain", [("Škofjeloško", .85), ("Škofjeloški", .75)], "hills W of Škofja Loka incl. Blegoš (hand-estimated)")
R("Polhograjsko hribovje", 46.08, 14.28, 13, 7, 100, "mountain", [("Polhograjski Dolomiti", 1), ("Polhograjsko", .9)], "GeoNames MTS 1374 Polhograjski Dolomiti (46.10,14.267)")
R("Velika planina", 46.30, 14.667, 5, 3.5, 90, "plateau", [], "GeoNames MT 422 (46.30,14.667)")
R("Logarska dolina", 46.40, 14.63, 5, 2, 0, "valley", [("Logarska", .9)], "GeoNames VAL 1984 (46.40,14.633)")
R("Solčavsko", 46.42, 14.67, 12, 8, 90, "region", [("Solčavski", .8)], "Logarska dolina, Robanov kot, Matkov kot, Solčava (hand-estimated)")
R("Zgornja Savinjska dolina", 46.33, 14.90, 14, 6, 95, "valley", [("Zgornjesavinjska dolina", 1), ("Zgornjesavinjska", .9), ("Zgornje Savinjska", .9)], "Luče-Ljubno-Mozirje-Nazarje (hand-estimated)")
R("Šaleška dolina", 46.36, 15.07, 8, 4, 90, "valley", [("Šaleški", .8)], "Velenje-Šoštanj-Šmartno ob Paki area (hand-estimated); GeoNames lakes Velenjsko/Šoštanjsko jezero (46.37,15.07)")
R("Savinjska dolina", 46.25, 15.15, 14, 7, 90, "valley", [("Spodnja Savinjska dolina", 1)], "Celje-Žalec-Polzela-Prebold (hand-estimated)")
R("Uršlja gora", 46.483, 14.925, 4, 3, 0, "mountain", [], "GeoNames PPL Uršlja Gora (46.50,14.933); peak Plešivec 1699 m")
R("Boč", 46.283, 15.60, 5, 3, 70, "mountain", [], "GeoNames MT 3249 (46.283,15.60)")
R("Donačka gora", 46.258, 15.735, 4, 2.5, 45, "mountain", [("Donačka", .9)], "GeoNames PPL Donačka Gora (46.258,15.735)")
R("Karavanke", 46.45, 14.35, 40, 7, 95, "mountain", [("Karavanški", .7)], "GeoNames MTS 2307 'Karawanken' (46.417,14.417); Slovenian side Jesenice-Dravograd")
R("Kamniško-Savinjske Alpe", 46.35, 14.60, 25, 10, 75, "mountain", [("Kamniške Alpe", 1), ("Savinjske Alpe", 1), ("Kamniško-Savinjske", .9)],
  "GeoNames MTS 2324 'Kamniške Planine' (46.25,14.667), re-centred to the range")
R("Julijske Alpe", 46.37, 13.80, 28, 14, 100, "mountain", [("Julijci", .6)], "GeoNames MTS 2360 'Julian Alps' (46.333,13.75), re-centred to the Slovene part")
R("Posočje", 46.25, 13.65, 25, 8, 150, "region", [("Dolina Soče", 1), ("Soška dolina", 1)], "Soča valley Bovec-Kobarid-Tolmin (hand-estimated)")
R("Vipavska dolina", 45.86, 13.93, 22, 5, 120, "valley", [("Vipavsko", .9)], "Ajdovščina-Vipava-Nova Gorica (hand-estimated)")
R("Čičarija", 45.52, 14.00, 12, 6, 135, "mountain", [], "hills S of Podgrad on the SI/HR border (hand-estimated)")
R("Slavnik", 45.532, 13.975, 4, 3, 0, "mountain", [], "hand-estimated (peak 1028 m)")
R("Zasavje", 46.14, 15.00, 12, 7, 100, "region", [("Zasavski", .8)], "Zagorje-Trbovlje-Hrastnik (hand-estimated)")
R("Posavje", 45.98, 15.45, 25, 10, 100, "region", [("Posavski", .75)], "Sevnica-Krško-Brežice (hand-estimated)")
R("Obsotelje", 46.18, 15.55, 10, 8, 90, "region", [], "Rogaška Slatina-Šmarje pri Jelšah (hand-estimated)")
R("Mežiška dolina", 46.52, 14.89, 11, 3, 45, "valley", [("Mežiško", .8)], "Črna-Mežica-Prevalje-Ravne (hand-estimated)")
R("Bohinj", 46.30, 13.90, 14, 7, 100, "region", [("Bohinjsko", .9), ("Bohinjski", .8)], "Bohinj valley incl. lake and Bohinjska Bistrica; GeoNames RGN 3240 coordinate (46.333,14.083) looks off")
R("Jezersko", 46.39, 14.50, 6, 4, 90, "area", [], "Zgornje/Spodnje Jezersko (GeoNames 46.394,14.507)")
R("Trenta", 46.39, 13.75, 8, 3, 150, "valley", [], "Trenta valley (GeoNames VAL 512 46.367,13.717)")
R("Tuhinjska dolina", 46.22, 14.75, 7, 3, 90, "valley", [], "GeoNames RGN 474 'Tuhinj Valley' (46.217,14.75)")
R("Golte", 46.374, 14.901, 4, 3, 90, "plateau", [], "GeoNames HTL 'Mozirska koča na Golteh' (46.374,14.901); karst plateau above Mozirje")
R("Cerkniško jezero", 45.74, 14.40, 5, 3, 0, "area", [("Cerkniško polje", 1)], "intermittent lake / polje between Cerknica and Otok (hand-estimated)")
R("Sviščaki", 45.575, 14.41, 3, 2.5, 0, "area", [], "forest hamlet/ski area on the NW foot of Snežnik (hand-estimated)")

# ---- additions (field-report gaps): places the labeler named that GeoNames lacks -----------------------------
# Centres and extents are approximate (+-3-5 km); where the place is a hamlet / hill range the ellipse is a "where
# this person goes" area, not a boundary.  Basis strings say how each one was placed.
R("Smrekovec", 46.42, 14.97, 7, 3.5, 70, "mountain", [("Smrekovško pogorje", 1), ("Smrekovca", .9)],
  "hand-estimated: peak 1577 m / Smrekovec range between Šoštanj, Mozirje and Šoštanj-Luče (forum: 'pogorje Smrekovca ali Lisca' near Sleme nad Šoštanjem)")
R("Resevna", 46.14, 15.41, 5, 3.5, 90, "area", [], "hand-estimated: hamlet area in Šentjur pri Celju on the N foot of Bohor (forum: 'Resevna - Bohor', 'Resevna (Šentjur pri Celju)'); +-4 km", weight=.85)
R("Košenjak", 46.60, 15.05, 5, 3, 80, "mountain", [], "hand-estimated: ridge 1522 m on the Austrian border N of Dravograd/Radlje (forum: 'Koroška proti Ojstrici - Košenjaku'); +-4 km")
R("Kobansko", 46.60, 15.13, 14, 6, 85, "area", [("Kobanski", .75)], "hand-estimated: hill country between Dravograd, Radlje ob Dravi and the Austrian border W of Pohorje; forum compares it with Pohorje")
R("Sorško polje", 46.21, 14.33, 6, 3.5, 10, "area", [], "hand-estimated: plain of the Sora river between Kranj, Škofja Loka and Medvode")
R("Jošt", 46.245, 14.304, 3.5, 2.5, 90, "mountain", [("Jošt nad Kranjem", 1), ("Sveti Jošt", 1)],
  "GeoNames PPL 'Sveti Jošt nad Kranjem' (46.245,14.304); wooded ridge above Kranj, Škofja Loka and Medvode", weight=.9)
R("Uskovnica", 46.325, 13.925, 3.5, 2.5, 90, "plateau", [], "hand-estimated: meadow plateau (~1100 m) on the S edge of Pokljuka above Bohinj; +-4 km")
R("Murska šuma", 46.60, 16.12, 14, 4, 110, "forest", [("Murski gozd", .8)],
  "hand-estimated, UNCERTAIN: floodplain forest along the Mura in Pomurje (Radenci - Veržej - Razkrižje); extent is the whole stretch")
R("Ljubljansko barje", 45.97, 14.43, 13.5, 6, 95, "area", [("Barje", .8), ("Ljubljanskem barju", 1)],
  "hand-estimated: marshy basin S of Ljubljana, Vrhnika - Ig - Brezovica (about 14.25-14.60 E, 45.90-46.03 N)")
R("Istra", 45.52, 13.83, 16, 9, 60, "region", [("Slovenska Istra", 1), ("Istrski", .7), ("Istrsko", .7)],
  "hand-estimated: SLOVENIAN Istria only (Koper - Izola - Piran and the hinterland to Črni Kal); the labeler's 'Istra' may include Croatian Istria, which cannot be located here", weight=.8)
R("Soriška planina", 46.215, 14.065, 4, 3, 90, "plateau", [("Soriška", .8)], "hand-estimated: ski plateau (~1300-1600 m) above Sorica, between Selška and Bohinj")
R("Ljubelj", 46.41, 14.27, 4, 2.5, 120, "mountain", [], "hand-estimated: Ljubelj pass and its forests on the SI/AT border above Tržič")
R("Mašun", 45.63, 14.37, 4, 3, 135, "forest", [("Mašunski gozd", 1)], "GeoNames PPL Mašun (45.628,14.368) + surrounding Snežnik-foot forest")
# Pohorje sub-areas: centres on the main ridge between Rogla and Areh, +-4 km (the whole-Pohorje ellipse is the fallback)
R("Šumik", 46.468, 15.305, 3.5, 2.5, 90, "area", [("Šumik Kapla", 1)], "hand-estimated: Šumik forest reserve / waterfall W of Rogla (Lovrenc na Pohorju side); +-4 km")
R("Osankarica", 46.485, 15.405, 3.5, 2.5, 90, "area", [], "hand-estimated: ridge hut and memorial on central Pohorje between Rogla and Areh; +-4 km")
R("Črno jezero", 46.455, 15.385, 3, 2.5, 90, "area", [("Črnega jezera", 1)], "hand-estimated: Black Lake peat bog on central Pohorje (~1300 m); +-4 km")
R("Klopni vrh", 46.50, 15.43, 3.5, 2.5, 90, "mountain", [], "hand-estimated: ridge on central/E Pohorje N of Osankarica; +-5 km")
R("Slivniško Pohorje", 46.46, 15.64, 7, 4, 80, "mountain", [("Hočko Pohorje", .9), ("Slivniško Hočko Pohorje", 1)], "hand-estimated: SE/E slopes of Pohorje above Hoče and Slivnica pri Mariboru")


CURATED = _R
