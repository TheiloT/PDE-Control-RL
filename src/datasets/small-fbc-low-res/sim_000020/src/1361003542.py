# This cell generates training dataset. If a constant dataset over sessions is desired, it should be run only once to generate the datasets, which should be zipped for later usage. Should not be confused with the training data used in the RL training process, which is generated on the go.
for batch_index in range(SCENE_COUNT // BATCH_SIZE):
    scene = Scene.create(DATA_PATH, count=BATCH_SIZE)  # Instanciates batch_size scenes
    print(scene)
    world = World()
    u0 = BurgersVelocity(
        DOMAIN,
        velocity=GaussianClash(BATCH_SIZE),  # TT: Initialize the velocity as the sum of two gaussians (a left one and a right one) of opposite amplitudes
        viscosity=VISCOSITY,
        batch_size=BATCH_SIZE,
        name='burgers'
    )
    u = world.add(u0, physics=Burgers(diffusion_substeps=DIFFUSION_SUBSTEPS))
    force = world.add(FieldEffect(GaussianForce(BATCH_SIZE), ['velocity']))
    # scene.write(world.state, frame=0)
    for frame in range(1, STEP_COUNT + 1):
        world.step(dt=DT)
        # scene.write(world.state, frame=frame)