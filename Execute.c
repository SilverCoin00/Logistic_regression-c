#include "D:\Data\code_doc\AI_model_building\Logistic_regression\Core.h"

int main() {
    char file[] = "D:\\Data\\archive1\\MAGIC_Gamma_Telescope_small_copy.csv";
    Data_Frame* df = read_csv(file, 1000, ",");
    Dataset* ds = trans_dframe_to_dset(df, "class");
    Standard_scaler* scaler = (Standard_scaler*)malloc(sizeof(Standard_scaler));
    scaler_fit(ds->x, NULL, ds->samples, ds->features, scaler, "Standard_scaler");
    scaler_transform(ds->x, NULL, ds->samples, ds->features, scaler, "Standard_scaler");
    //print_dataset(ds, 5, 10, 15);
    Logistic_Regression* model = (Logistic_Regression*)malloc(sizeof(Logistic_Regression));
    model->weights = init_weights(ds->features, 5);
    model->data = ds;
    train(model, "NAG", 250, 1e-3, 32);
    free_lg_model(model);
    free_scaler(scaler, "Standard_scaler");
    free_data_frame(df);
    free_dataset(ds);
    return 0;
}