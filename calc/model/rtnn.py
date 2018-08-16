from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from calc.model.tnn import *


def simple_train(tnn_model, tx, ty, vx=None, vy=None, epochs=300, batch_size=1024, model_save_dir='.', model_id='rtnn',
                 lr_patience=30, stop_patience=50):
    checkpointer = ModelCheckpoint(os.path.join(model_save_dir, model_id), monitor='val_loss', verbose=1,
                                   save_best_only=True, save_weights_only=True, period=1)
    checkpointer_reboot = ModelCheckpoint(os.path.join(model_save_dir, f'{model_id}_reboot'), monitor='val_loss',
                                          verbose=1, save_best_only=False, save_weights_only=True, period=1)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=lr_patience, verbose=1, min_lr=1e-5)
    early_stopper = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=stop_patience, verbose=1)

    tnn_model.fit(tx, ty, epochs=epochs, batch_size=batch_size, validation_data=(vx, vy) if vx is not None else None,
                  verbose=2, callbacks=[checkpointer_reboot, checkpointer, lr_scheduler, early_stopper])

    return read_weights(tnn_model, os.path.join(model_save_dir, model_id))


def train(tx, ty, tnn_models, train_func=simple_train, vx=None, vy=None, res_shrinkage=0.1, predict_batch_size=50000,
          **train_params):
    rtnns = []
    for tnn_model in tnn_models:
        rtnn = train_func(tnn_model, tx, ty, vx, vy, **train_params)
        rtnns.append(rtnn)

        tp = np.squeeze(rtnn.predict(tx, batch_size=predict_batch_size))
        vp = np.squeeze(rtnn.predict(vx, batch_size=predict_batch_size))
        ty -= res_shrinkage * tp
        vy -= res_shrinkage * vp

    return rtnns


def predict(x, rtnns, res_shrinkage, predict_batch_size=50000):
    p = None
    if rtnns:
        p = np.squeeze(rtnns[-1].predict(x, batch_size=predict_batch_size))
        for rtnn in rtnns[:-1]:
            pp = np.squeeze(rtnn.predict(x, batch_size=predict_batch_size))
            p += res_shrinkage * pp
    return p
