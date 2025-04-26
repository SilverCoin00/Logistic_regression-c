#include "D:\Data\code_doc\AI_model_building\Logistic_regression\Core.h"

typedef struct Weights {
    float* weights;
	int num_weights;
} Weights;

Weights* init_weights(int num_of_features, int random_init) {
	Weights* newb = (Weights*)malloc(sizeof(Weights));
	newb->num_weights = num_of_features + 1;
	newb->weights = (float*)calloc(num_of_features + 1, sizeof(float));
	srand(random_init);
	for (int i = 0; i < newb->num_weights; i++) newb->weights[i] = ((float)rand() / RAND_MAX)* 0.2 - 0.1;
	return newb;
}
float* weights_derivative(Dataset* data, Weights* w) {             // deriv(w) = (1 / m).X(T).(1 / (1 + exp(- X.w)) - y)
	int i;
	float* deriv = (float*)malloc(w->num_weights* sizeof(float));
	float** weights = new_matrix(w->num_weights, 1);
	for (i = 0; i < w->num_weights; i++) weights[i][0] = w->weights[i];
	float** error = matrix_multiply(data->x, weights, data->samples, data->features + 1, 1);
	free_matrix(weights, w->num_weights);

	for (i = 0; i < data->samples; i++) error[i][0] = 1 / (1 + exp(- error[i][0])) - data->y[i];
	
	float** x_T = transpose_matrix(data->x, data->samples, data->features + 1);
	float** t = matrix_multiply(x_T, error, data->features + 1, data->samples, 1);
	free_matrix(x_T, data->features + 1);
	free_matrix(error, data->samples);
	for (i = 0; i < data->features + 1; i++) {
		deriv[i] = t[i][0] / data->samples;
		free(t[i]);
	}
	free(t);
	return deriv;
}
void print_weights(Weights* w, int decimal) {
	printf("Weights: [");
	for (int i = 0; i < w->num_weights - 1; i++) printf("%.*f, ", decimal, w->weights[i]);
	printf("%.*f]\n", decimal, w->weights[w->num_weights - 1]);
}
void free_weights(Weights* w) {
	free(w->weights);
	free(w);
}
void grad_descent(Dataset* data, Weights* w, float learning_rate) {
	float* gradient = weights_derivative(data, w);
	for (int i = 0; i < w->num_weights; i++) w->weights[i] -= learning_rate* gradient[i];
	free(gradient);
}
void grad_descent_momentum(Dataset* data, Weights* w, float learning_rate, float* pre_velocity, float velocity_rate) {
	float* velo = weights_derivative(data, w);
	for (int i = 0; i < w->num_weights; i++) {
		velo[i] *= learning_rate;
		velo[i] += velocity_rate* pre_velocity[i];
		w->weights[i] -= velo[i];
		pre_velocity[i] = velo[i];
	}
	free(velo);
}
void nesterov_accelerated_grad(Dataset* data, Weights* w, float learning_rate, float* pre_velocity, float velocity_rate) {
	Weights* fore_w = (Weights*)malloc(sizeof(Weights));
	fore_w->num_weights = w->num_weights;
	fore_w->weights = (float*)malloc(fore_w->num_weights* sizeof(float));
	int i;
	for (i = 0; i < w->num_weights; i++) fore_w->weights[i] = w->weights[i] - velocity_rate* pre_velocity[i];
	float* velo = weights_derivative(data, fore_w);
	free_weights(fore_w);
	for (i = 0; i < w->num_weights; i++) {
		velo[i] *= learning_rate;
		velo[i] += velocity_rate* pre_velocity[i];
		w->weights[i] -= velo[i];
		pre_velocity[i] = velo[i];
	}
	free(velo);
}