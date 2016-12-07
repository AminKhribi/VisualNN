These are scripts that compose a Neural Net with Keras, and allows to fit and plot the architecture dynamically while learning.

How to use in a python script:

create_model = functools.partial(models_gen.create_model_gen, n_layers=n_layers,
                                                               input_dim=input_dim,
                                                               K=K,  # list of number of neurones in layers
                                                               D=D,  # list of dropout coef in layers
                                                               act=act,
                                                               init=init,
                                                               optimizer=optimizer
                                                               )

param_model = dict(
              optimizer=optimizer,
              K=random.choice(K_grid),
              D=random.choice(D_grid),
              w_cons=random.choice(w_cons_grid),
              l2=random.choice(l2_grid),
              l1=random.choice(l1_grid)
            )

param_fit = dict(batch_size=batch_size,
                 nb_epoch=nb_epoch,
                 validation_split=validation_split,
                 patience=patience)

score, model = computation.compute_nn(create_model=create_model,
                                      param_model=param_model,
                                      param_fit=param_fit,
                                      X_learn=X_learn_nn,
                                      Y_learn=Y_learn,
                                      X_test=X_test_nn,
                                      Y_test=Y_test,
                                      plot=True)


![alt tag](https://github.com/AminKhribi/VisualNN/blob/master/nn.png)
