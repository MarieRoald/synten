from synten import timeseries


def test_timeseries_component():
    for name, Component in timeseries.time_factor_register.items():
        for num_timesteps in [10, 50, 100]:
            print(f"Testing {name}")
            assert len(Component().generate_factor(num_timesteps)) == num_timesteps, f"Length of {name} is incorrect"


def test_timeseries_factor_generator():
    for name in timeseries.time_factor_register:
        for num_timesteps in [10, 50, 100]:
            print(f"Testing {name}")
            generator = timeseries.TimeSeriesFactorGenerator(
                num_timesteps=num_timesteps,
                components=[{"type": name}]
            )
            assert len(generator.generate_factors()) == num_timesteps


            