#include "mpi.h"
#include <cstdio>
#include <cstdlib>
#define NRA 100
#define NCA 100
#define NCB 100
#define MASTER 0

double** alloc_dynamic_array(int rows, int cols) {
    auto* mem = (double*)malloc(rows * cols * sizeof(double));
    auto** A = (double**)malloc(rows * sizeof(double*));
    A[0] = mem;
    for (int i = 1; i < rows; i++) A[i] = A[i - 1] + cols;
    return A;
}

void empty_dynamic_array(double** arr) {
    free(arr[0]);
    free(arr);
}

int main(int argc, char* argv[]) {
    int numtasks,
        taskid,
        numworkers,
        i, j, rc = 0;

    double time;

    double** a = alloc_dynamic_array(NRA, NCA);
    double** b = alloc_dynamic_array(NCA, NCB);
    double** c = alloc_dynamic_array(NRA, NCB);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);

    double** a_rows = alloc_dynamic_array(NRA / numtasks, NRA);
    double** c_rows = alloc_dynamic_array(NRA / numtasks, NRA);

    if (numtasks < 2) {
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }
    numworkers = numtasks - 1;

    if (taskid == MASTER) {
        printf("mpi_mm has started with %d tasks.\n", numtasks);

        for (i = 0; i < NRA; i++)
            for (j = 0; j < NCA; j++)
                a[i][j] = 10;
        for (i = 0; i < NCA; i++)
            for (j = 0; j < NCB; j++)
                b[i][j] = 10;

        time = MPI_Wtime();
    }

    MPI_Scatter(
            *a,
            NRA * NRA / numtasks,
            MPI_DOUBLE,
            *a_rows,
            NRA * NRA / numtasks,
            MPI_DOUBLE,
            MASTER,
            MPI_COMM_WORLD);

    MPI_Bcast(
            *b,
            NRA * NRA,
            MPI_DOUBLE,
            MASTER,
            MPI_COMM_WORLD);

    for (int k = 0; k < NRA / numtasks; k++) {
        for (int m = 0; m < NRA; m++) {
            c_rows[k][m] = 0.0e0;
        }
    }

    for (int d = 0; d < NRA / numtasks; d++) {
        for (int k = 0; k < NRA; k++) {
            for (int f = 0; f < NRA; f++) {
                c_rows[d][f] += a_rows[d][k] * b[k][f];
            }
        }
    }

    MPI_Gather(
            *c_rows,
            (NRA / numtasks) * NRA,
            MPI_DOUBLE,
            *c,
            (NRA / numtasks) * NRA,
            MPI_DOUBLE,
            MASTER,
            MPI_COMM_WORLD);

    if (taskid == MASTER) {
        time = MPI_Wtime() - time;

        printf("\nTotal time: %.2f\n", time);
    }

    empty_dynamic_array(a);
    empty_dynamic_array(b);
    empty_dynamic_array(c);
    empty_dynamic_array(a_rows);
    empty_dynamic_array(c_rows);

    MPI_Finalize();
} 