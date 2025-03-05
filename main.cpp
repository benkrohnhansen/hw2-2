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

    double ghost_calc; double time_ghost_cnt_comm; double ghost_cnt_waitall; 
    double ghost_comm; double ghost_waitall; double force_calc; double move; 
    double calc_particle_mv; double part_cnt_waitall; double part_cnt; 
    double part_waitall; double part; double part_insert; double end_barrier;

    double sort_time; double gather_time;

    for (int step = 0; step < nsteps; ++step) {
        simulate_one_step(parts, num_parts, size, rank, num_procs,
            ghost_calc, time_ghost_cnt_comm, ghost_cnt_waitall, 
            ghost_comm, ghost_waitall, force_calc, move, 
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
    if (rank == 0) {
        std::cout << "Simulation Time = " << seconds << " seconds for " << num_parts
                  << " particles.\n\n";
        std::cout << "==== Simulation Timing Breakdown ====" << std::endl;
        std::cout << "Ghost Particle Calculation Time: " << ghost_calc << " seconds" << std::endl;
        std::cout << "Ghost Particle Count Comm Time: " << time_ghost_cnt_comm << " seconds" << std::endl;
        std::cout << "Ghost Count Waitall Time: " << ghost_cnt_waitall << " seconds" << std::endl;
        std::cout << "Ghost Particle Comm Time: " << ghost_comm << " seconds" << std::endl;
        std::cout << "Ghost Waitall Time: " << ghost_waitall << " seconds" << std::endl;
        std::cout << "Force Calculation Time: " << force_calc << " seconds" << std::endl;
        std::cout << "Particle Move Time: " << move << " seconds" << std::endl;
        std::cout << "Calculate Particle Movement Time: " << calc_particle_mv << " seconds" << std::endl;
        std::cout << "Particle Count Waitall Time: " << part_cnt_waitall << " seconds" << std::endl;
        std::cout << "Particle Count Time: " << part_cnt << " seconds" << std::endl;
        std::cout << "Particle Waitall Time: " << part_waitall << " seconds" << std::endl;
        std::cout << "Particle Exchange Time: " << part << " seconds" << std::endl;
        std::cout << "Particle Insert Time: " << part_insert << " seconds" << std::endl;
        std::cout << "End Barrier Time: " << end_barrier << " seconds" << std::endl;
        std::cout << "Sorting Time: " << sort_t << " seconds" << std::endl;
        std::cout << "Gathering Time: " << gather_t << " seconds" << std::endl;
        std::cout << "=====================================" << std::endl;
    }
    if (fsave) {
        fsave.close();
    }
    delete[] parts;

    MPI_Type_free(&PARTICLE);
    MPI_Finalize();
}
