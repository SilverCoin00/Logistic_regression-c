#pragma once
#include "D:\Data\code_doc\AI_model_building\Pandas_&_Numpy.c"
#include "D:\Data\code_doc\AI_model_building\Sklearn.c"

typedef struct Dataset {
	float** x;
	float* y;
	int features;      // x_cols
	int samples;       // rows
} Dataset;

Dataset* new_dataset(float** x_train, float* y_train, int num_of_features, int num_of_samples) {
	Dataset* newd = (Dataset*)malloc(sizeof(Dataset));
	newd->x = (float**)malloc(num_of_samples* sizeof(float*));
	for (int i = 0; i < num_of_samples; i++) {
		newd->x[i] = (float*)malloc((num_of_features + 1)* sizeof(float));
		for (int j = 0; j < num_of_features; j++) newd->x[i][j] = x_train[i][j];
		newd->x[i][num_of_features] = 1;
	}
	newd->y = (float*)malloc(num_of_samples* sizeof(float));
	for (int i = 0; i < num_of_samples; i++) newd->y[i] = y_train[i];
	newd->features = num_of_features;
	newd->samples = num_of_samples;
	return newd;
}
Dataset* trans_dframe_to_dset(Data_Frame* df, const char* predict_feature_col) {
	float** enc_sdata;
	int y_col = strtoi(predict_feature_col), i, j, k, is_y_str = 0;
	if (df->str_cols[0] != 0) {
		enc_sdata = (float**)malloc(df->str_cols[0]* sizeof(float*));
		Label_encoder* encoder = (Label_encoder*)malloc(sizeof(Label_encoder));
		for (i = 0; i < df->str_cols[0]; i++) {
			encoder_fit(df->str_data, df->row, i, encoder, "Label_encoder");
			enc_sdata[i] = (float*) encoder_transform(df->str_data, df->row, i, encoder, "Label_encoder");
			free_set(encoder->sample_types);
		}
		free(encoder);
	}
	if (y_col < 0) {
		j = df->col + df->str_cols[0];
		for (i = 0; i < j; i++) {
			if (!strcmp(df->features[i], predict_feature_col)) {
				for (k = 1; k <= df->str_cols[0]; k++)
					if (df->str_cols[k] == i) {
						y_col = k - 1;
						is_y_str = 1;
						goto out;
					}
				y_col = i;
				break;
			}
		}
	}
	out:;
	Dataset* newd = (Dataset*)malloc(sizeof(Dataset));
	newd->y = (float*)calloc(df->row, sizeof(float));
	if (is_y_str) for (i = 0; i < df->row; i++) newd->y[i] = (float) enc_sdata[y_col][i];
	else {
		if (y_col >= 0) for (i = 0; i < df->row; i++) newd->y[i] = df->data[i][y_col];
	}
	newd->x = (float**)malloc(df->row* sizeof(float*));
	for (i = 0; i < df->row; i++) {
		newd->x[i] = (float*)malloc((df->col + df->str_cols[0])* sizeof(float));  // drop 1 col for y but plus 1 for bias, so, nothing changes
		for (j = 0, k = 0; j < df->col; k++) {
			if (!is_y_str) if (k == y_col) continue;
			newd->x[i][j++] = df->data[i][k];
		}
		for (k = 0; k < df->str_cols[0]; k++) {
			if (is_y_str) if (k == y_col) continue;
			newd->x[i][j++] = enc_sdata[k][i];
		}
		newd->x[i][df->col + df->str_cols[0] - 1] = 1;
	}
	newd->features = df->col + df->str_cols[0] - 1;
	newd->samples = df->row;
	if (enc_sdata) free_matrix(enc_sdata, df->str_cols[0]);
	return newd;
}
Dataset* dataset_copy(const Dataset* ds) {
	Dataset* newd = (Dataset*)malloc(sizeof(Dataset));
	newd->x = (float**)malloc(ds->samples* sizeof(float*));
	for (int i = 0; i < ds->samples; i++) {
		newd->x[i] = (float*)malloc((ds->features + 1)* sizeof(float));
		for (int j = 0; j < ds->features; j++) newd->x[i][j] = ds->x[i][j];
		newd->x[i][ds->features] = 1;
	}
	newd->y = (float*)malloc(ds->samples* sizeof(float));
	for (int i = 0; i < ds->samples; i++) newd->y[i] = ds->y[i];
	newd->features = ds->features;
	newd->samples = ds->samples;
	return newd;
}
void dataset_sample_copy(const Dataset* ds, int ds_sample_index, Dataset* copy, int copy_sample_index) {
	for (int i = 0; i < ds->features && i < copy->features; i++)
		copy->x[copy_sample_index][i] = ds->x[ds_sample_index][i];
	copy->x[copy_sample_index][copy->features] = 1;
	copy->y[copy_sample_index] = ds->y[ds_sample_index];
}
Dataset* dataset_samples_copy(const Dataset* ds, int ds_begin_index, int ds_end_index) {
	Dataset* newd = (Dataset*)malloc(sizeof(Dataset));
	newd->x = (float**)malloc((ds_end_index - ds_begin_index)* sizeof(float*));
	int i, j;
	for (i = ds_begin_index; i < ds_end_index && i < ds->samples; i++) {
		newd->x[i - ds_begin_index] = (float*)malloc((ds->features + 1)* sizeof(float));
		for (j = 0; j < ds->features; j++) newd->x[i - ds_begin_index][j] = ds->x[i][j];
		newd->x[i - ds_begin_index][ds->features] = 1;
	}
	newd->y = (float*)malloc((ds_end_index - ds_begin_index)* sizeof(float));
	for (i = ds_begin_index; i < ds_end_index && i < ds->samples; i++) newd->y[i - ds_begin_index] = ds->y[i];
	newd->features = ds->features;
	newd->samples = ds_end_index - ds_begin_index;
	return newd;
}
Dataset* dataset_samples_order_copy(const Dataset* ds, int* order, int order_begin_index, int order_end_index) {
	Dataset* newd = (Dataset*)malloc(sizeof(Dataset));
	newd->x = (float**)malloc((order_end_index - order_begin_index)* sizeof(float*));
	newd->y = (float*)malloc((order_end_index - order_begin_index)* sizeof(float));
	newd->features = ds->features;
	newd->samples = order_end_index - order_begin_index;
	for (int i = order_begin_index; i < order_end_index; i++) {
		newd->x[i - order_begin_index] = (float*)malloc((ds->features + 1)* sizeof(float));
		dataset_sample_copy(ds, order[i], newd, i - order_begin_index);
	}
	return newd;
}
void print_dataset(Dataset* ds, int decimal, int col_space, int num_of_rows) {
	if (!ds) return ;
	if (num_of_rows < 0 || num_of_rows > ds->samples) num_of_rows = ds->samples;
	printf(" Row\n");
	for (int i = 0, j; i < num_of_rows; i++) {
		printf("%4d\t", i + 1);
		for (j = 0; j < ds->features; j++) {
			printf("%*.*f ", col_space, decimal, ds->x[i][j]);
		}
		printf("\t| %*.*f\n", col_space, decimal, ds->y[i]);
	}
}
void free_dataset(Dataset* ds) {
	if (!ds) return ;
	for (int i = 0; i < ds->samples; i++) free(ds->x[i]);
	free(ds->x);
	free(ds->y);
	free(ds);
}