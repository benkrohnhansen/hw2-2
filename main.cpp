#include "common.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <random>
#include <vector>

// =================
// Helper Functions
// =================

// I/O routines
void save(std::ofstream& fsave, particle_t* parts, int num_parts, double size) {
    static bool first = true;

    if (first) {
        fsave << num_parts << " " << size << "\n";
        first = false;
    }

    for (int i = 0; i < num_parts; ++i) {
        fsave << parts[i].x << " " << parts[i].y << "\n";
    }

    fsave << std::endl;
}

// Particle Initialization
void init_particles(particle_t* parts, int num_parts, double size, int part_seed) {
    std::random_device rd;
    std::mt19937 gen(part_seed ? part_seed : rd());

    int sx = (int)ceil(sqrt((double)num_parts));
    int sy = (num_parts + sx - 1) / sx;

    std::vector<int> shuffle(num_parts);
    for (int i = 0; i < shuffle.size(); ++i) {
        shuffle[i] = i;
    }

    for (int i = 0; i < num_parts; ++i) {
        // Make sure particles are not spatially sorted
        std::uniform_int_distribution<int> rand_int(0, num_parts - i - 1);
        int j = rand_int(gen);
        int k = shuffle[j];
        shuffle[j] = shuffle[num_parts - i - 1];

        // Distribute particles evenly to ensure proper spacing
        parts[i].x = size * (1. + (k % sx)) / (1 + sx);
        parts[i].y = size * (1. + (k / sx)) / (1 + sy);

        // Assign random velocities within a bound
        std::uniform_real_distribution<float> rand_real(-1.0, 1.0);
        parts[i].vx = rand_real(gen);
        parts[i].vy = rand_real(gen);
    }

    for (int i = 0; i < num_parts; ++i) {
        parts[i].id = i + 1;
    }
}

// Command Line Option Processing
int find_arg_idx(int argc, char** argv, const char* option) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], option) == 0) {
            return i;
        }
    }
    return -1;
}

int find_int_arg(int argc, char** argv, const char* option, int default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return std::stoi(argv[iplace + 1]);
    }

    return default_value;
}

char* find_string_option(int argc, char** argv, const char* option, char* default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return argv[iplace + 1];
    }

    return default_value;
}

MPI_Datatype PARTICLE;

// ==============
// Main Function
// ==============

int main(int argc, char** argv) {
    // Parse Args
    if (find_arg_idx(argc, argv, "-h") >= 0) {
        std::cout << "Options:" << std::endl;
        std::cout << "-h: see this help" << std::endl;
        std::cout << "-n <int>: set number of particles" << std::endl;
        std::cout << "-o <filename>: set the output file name" << std::endl;
        std::cout << "-s <int>: set particle initialization seed" << std::endl;
        return 0;
    }

    // Open Output File
    char* savename = find_string_option(argc, argv, "-o", nullptr);
    std::ofstream fsave(savename);

    // Init MPI
    int num_procs, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Create MPI Particle Type
    const int nitems = 7;
    int blocklengths[7] = {1, 1, 1, 1, 1, 1, 1};
    MPI_Datatype types[7] = {MPI_UINT64_T, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,
                             MPI_DOUBLE,   MPI_DOUBLE, MPI_DOUBLE};
    MPI_Aint offsets[7];
    offsets[0] = offsetof(particle_t, id);
    offsets[1] = offsetof(particle_t, x);
    offsets[2] = offsetof(particle_t, y);
    offsets[3] = offsetof(particle_t, vx);
    offsets[4] = offsetof(particle_t, vy);
    offsets[5] = offsetof(particle_t, ax);
    offsets[6] = offsetof(particle_t, ay);
    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &PARTICLE);
    MPI_Type_commit(&PARTICLE);

    // Initialize Particles
    int num_parts = find_int_arg(argc, argv, "-n", 1000);
    int part_seed = find_int_arg(argc, argv, "-s", 0);
    double size = sqrt(density * num_parts);

    particle_t* parts = new particle_t[num_parts];

    if (rank == 0) {
        init_particles(parts, num_parts, size, part_seed);
    }

    MPI_Bcast(parts, num_parts, PARTICLE, 0, MPI_COMM_WORLD);

    // Algorithm
    auto start_time = std::chrono::steady_clock::now();

    init_simulation(parts, num_parts, size, rank, num_procs);
    
    auto end_init_time = std::chrono::steady_clock::now();

    double ghost_calc = 0.0, time_ghost_cnt_comm = 0.0, ghost_cnt_waitall = 0.0;
    double ghost_comm = 0.0, ghost_waitall = 0.0, force_calc = 0.0, move_time = 0.0;
    double calc_particle_mv = 0.0, part_cnt_waitall = 0.0, part_cnt = 0.0;
    double part_waitall = 0.0, part = 0.0, part_insert = 0.0, end_barrier = 0.0;
    double sort_time = 0.0, gather_time = 0.0;
    

    for (int step = 0; step < nsteps; ++step) {
        simulate_one_step(parts, num_parts, size, rank, num_procs,
            ghost_calc, time_ghost_cnt_comm, ghost_cnt_waitall, 
            ghost_comm, ghost_waitall, force_calc, move_time, 
            calc_particle_mv, part_cnt_waitall, part_cnt, 
            part_waitall, part, part_insert, end_barrier);

        // Save state if necessary
        if (fsave.good() && (step % savefreq) == 0) {
            gather_for_save(parts, num_parts, size, rank, num_procs, sort_time, gather_time);
            if (rank == 0) {
                save(fsave, parts, num_parts, size);
            }
        }
    }

    auto end_time = std::chrono::steady_clock::now();

    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();

    // Finalize
    if (rank == 1) {

    

        double comm_time = (time_ghost_cnt_comm - ghost_cnt_waitall) + (ghost_comm - ghost_waitall) 
                    + (part_cnt - part_cnt_waitall) + (part - part_waitall);
        double synch_time = part_cnt_waitall + ghost_waitall + part_waitall;
        double force_calc_time = force_calc;

        std::chrono::duration<double> init_diff = end_init_time - start_time;
	double init_time = init_diff.count();
        double other = seconds - (comm_time + synch_time + force_calc_time + init_time + end_barrier);
        
        std::cout << "\n==== Simulation Summary ====\n";
        std::cout << "Processors: " << num_procs << " | Rank: " << rank << " | Particles: " << num_parts << "\n\n";
        
        std::cout << "| Time Category     | Value (s)          |\n";
        std::cout << "|-------------------|-------------------|\n";
        std::cout << "| Isend/Irecv       | " << comm_time << " |\n";
        std::cout << "| Waitall           | " << synch_time << " |\n";
        std::cout << "| Force Calculation | " << force_calc_time << " |\n";
	std::cout << "| End Barrier       | " << end_barrier << " |\n";
        std::cout << "| Initialization    | " << init_time << " |\n";
        std::cout << "| Other             | " << other << " |\n";
        std::cout << "| Total             | " << seconds << " |\n";
        std::cout << "=========================================\n";

    }
    if (fsave) {
        fsave.close();
    }
    delete[] parts;

    MPI_Type_free(&PARTICLE);
    MPI_Finalize();
}
