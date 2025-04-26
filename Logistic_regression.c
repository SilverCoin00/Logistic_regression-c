#include "Core.h"

typedef struct Logistic_Regression {
    Dataset* data;
    Weights* weights;
} Logistic_Regression;

void predict(Dataset* data, Weights* w, float* y_pred) {
    if (!y_pred) return ;
    float** weights = new_matrix(w->num_weights, 1);
	for (int i = 0; i < w->num_weights; i++) weights[i][0] = w->weights[i];
	float** s = matrix_multiply(data->x, weights, data->samples, data->features + 1, 1);
	free_matrix(weights, w->num_weights);
	for (int i = 0; i < data->samples; i++) {
		y_pred[i] = 1 / (1 + exp(- s[i][0]));
		free(s[i]);
	}
    free(s);
}
float loss_func(float* y_pred, float* y_true, int length) {
	float val_loss = 0.0, pred;
	for (int i = 0; i < length; i++) {
		pred = y_pred[i];
		pred = pred > 1e-8 ? pred : 1e-8;
		pred = pred < 1 - 1e-8 ? pred : 1 - 1e-8;
		val_loss -= y_true[i]* log(pred) + (1 - y_true[i])* log(1 - pred);
	}
	return val_loss / length;
}
void train(Logistic_Regression* model, char* GD_type, int iteration, float learning_rate, int batch_size) {
	float* y_pred = (float*)malloc(model->data->samples* sizeof(float));
	float loss;
	float* pre_velo = (float*)calloc(model->weights->num_weights, sizeof(float));
	int* random_i = (int*)malloc(model->data->samples* sizeof(int)), i, loop;
	for (i = 0; i < model->data->samples; i++) random_i[i] = i;
	Dataset* batch;

	if (batch_size <= 0 || batch_size >= model->data->samples) batch_size = model->data->samples;
	else shuffle_index(random_i, model->data->samples, i);
	loop = model->data->samples / batch_size;

	while (iteration > 0) {
		predict(model->data, model->weights, y_pred);
		shuffle_index(random_i, model->data->samples, i);
		
		for (i = 0; i < loop; i++) {
			batch = dataset_samples_order_copy(model->data, random_i, i* batch_size, (i + 1)* batch_size);

			if (!strcmp(GD_type, "GD")) grad_descent(batch, model->weights, learning_rate);
			else if (!strcmp(GD_type, "GDM")) grad_descent_momentum(batch, model->weights, learning_rate, pre_velo, 0.9);
			else if (!strcmp(GD_type, "NAG")) nesterov_accelerated_grad(batch, model->weights, learning_rate, pre_velo, 0.9);
			free_dataset(batch);
		}
		if (batch_size* loop < model->data->samples) {
			batch = dataset_samples_order_copy(model->data, random_i, i* batch_size, model->data->samples);

			if (!strcmp(GD_type, "GD")) grad_descent(batch, model->weights, learning_rate);
			else if (!strcmp(GD_type, "GDM")) grad_descent_momentum(batch, model->weights, learning_rate, pre_velo, 0.9);
			else if (!strcmp(GD_type, "NAG")) nesterov_accelerated_grad(batch, model->weights, learning_rate, pre_velo, 0.9);
			free_dataset(batch);
		}
		loss = loss_func(y_pred, model->data->y, model->data->samples);
		printf("Iteration left: %d, loss = %.6f\n", iteration, loss);
		print_weights(model->weights, 8);
		iteration--;
	}
	free(random_i);
	free(pre_velo);
	free(y_pred);
}
void free_lg_model(Logistic_Regression* model) {
	if (model->data) free_dataset(model->data);
	if (model->weights) free_weights(model->weights);
	free(model);
}
