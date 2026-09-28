transport_cost = {
    "passenger_car_transport": {  # Burke et al. 2024 https://www.sciencedirect.com/science/article/pii/S0739885924000350
        "2020": {
            "fcev": {
                "inv": 50820,
                "om_var": 0.021,
                "om_fix": 0.0,
            },  # EUR2015/vehicle ; EUR2015/vehicle/year; EUR2015/km
            "ev": {"inv": 33202, "om_var": 0.008, "om_fix": 0.0},
            "icev": {"inv": 18232, "om_var": 0.027, "om_fix": 0.0},
        },
        "2030": {
            "fcev": {"inv": 25960, "om_var": 0.021, "om_fix": 0.0},
            "ev": {"inv": 22582, "om_var": 0.008, "om_fix": 0.0},
            "icev": {"inv": 18232, "om_var": 0.027, "om_fix": 0.0},
        },
        "2050": {
            "fcev": {"inv": 20210, "om_var": 0.021, "om_fix": 0.0},
            "ev": {"inv": 19832, "om_var": 0.008, "om_fix": 0.0},
            "icev": {"inv": 18232, "om_var": 0.027, "om_fix": 0.0},
        },
    },
    "light_duty_transport": {  # Burke et al. 2024 (pick-up truck)
        "2020": {
            "fcev": {"inv": 69987, "om_var": 0.021, "om_fix": 0.0},
            "ev": {"inv": 38284, "om_var": 0.008, "om_fix": 0.0},
            "icev": {"inv": 24424, "om_var": 0.027, "om_fix": 0.0},
        },
        "2030": {
            "fcev": {"inv": 34344, "om_var": 0.021, "om_fix": 0.0},
            "ev": {"inv": 27088, "om_var": 0.008, "om_fix": 0.0},
            "icev": {"inv": 24424, "om_var": 0.027, "om_fix": 0.0},
        },
        "2050": {
            "fcev": {"inv": 27360, "om_var": 0.021, "om_fix": 0.0},
            "ev": {"inv": 24249, "om_var": 0.0008, "om_fix": 0.0},
            "icev": {"inv": 24424, "om_var": 0.027, "om_fix": 0.0},
        },
    },
    "bus_transport": {
        "2020": {
            "fcev": {"inv": 404888, "om_var": 0.103576, "om_fix": 6026.24},
            "ev": {"inv": 433136, "om_var": 0.103576, "om_fix": 6026.24},
            "icev": {"inv": 159507, "om_var": 0.112992, "om_fix": 6779.52},
        },
        "2030": {
            "fcev": {"inv": 341809, "om_var": 0.103576, "om_fix": 6026.24},
            "ev": {"inv": 235400, "om_var": 0.103576, "om_fix": 6026.24},
            "icev": {"inv": 159507, "om_var": 0.112992, "om_fix": 6779.52},
        },
        "2050": {
            "fcev": {"inv": 201632, "om_var": 0.103576, "om_fix": 6026.24},
            "ev": {"inv": 201632, "om_var": 0.103576, "om_fix": 6026.24},
            "icev": {"inv": 159507, "om_var": 0.112992, "om_fix": 6779.52},
        },
    },  # EUR2015/vehicle source: DEA - B1
    "motorcycle_transport": {
        "2020": {
            "ev": {"inv": 20000.0, "om_var": 0.0, "om_fix": 0.0},  # own assumption
            "icev": {"inv": 15000.0, "om_var": 0.0, "om_fix": 0.0},
        },
        "2030": {
            "ev": {"inv": 14500.0, "om_var": 0.0, "om_fix": 0.0},  # own assumption
            "icev": {"inv": 15000.0, "om_var": 0.0, "om_fix": 0.0},
        },
        "2050": {
            "ev": {"inv": 12900.0, "om_var": 0.0, "om_fix": 0.0},
            "icev": {"inv": 15000.0, "om_var": 0.0, "om_fix": 0.0},
        },
    },
    "heavy_duty_transport": {
        "2020": {
            "fcev": {"inv": 465750, "om_var": 0.0, "om_fix": 20424},
            "ev": {"inv": 315330, "om_var": 0.0, "om_fix": 14617},
            "icev": {"inv": 121440, "om_var": 0.1129920, "om_fix": 20424},
        },
        "2030": {
            "fcev": {"inv": 155940, "om_var": 0.0, "om_fix": 15180},
            "ev": {"inv": 153180, "om_var": 0.0, "om_fix": 14617},
            "icev": {"inv": 132480, "om_var": 0.0, "om_fix": 20424},
        },
        "2050": {
            "fcev": {"inv": 140070, "om_var": 0.0, "om_fix": 15180},
            "ev": {"inv": 139380, "om_var": 0.0, "om_fix": 14617},
            "icev": {"inv": 132480, "om_var": 0.0, "om_fix": 20424},
        },
    },  # EUR2015/vehicle- 5-LH(800); convertion EUR23->EUR2015: 0.69, source: ICCT(2023) https://theicct.org/publication/total-cost-ownership-trucks-europe-nov23/
}
