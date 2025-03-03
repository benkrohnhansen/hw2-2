#include "common.h"
#include <mpi.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <cstring>
#include <algorithm>
// Put any static global variables here that you will use throughout the simulation.

// Apply force between two particles
void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);  // Ensure minimum distance between particles
    double r = sqrt(r2);  // Compute distance between particles

    // Short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

void apply_force(particle_t& particle, double& neighbor_x, double& neighbor_y) {
    // Calculate Distance
    double dx = neighbor_x - particle.x;
    double dy = neighbor_y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);  // Ensure minimum distance between particles
    double r = sqrt(r2);  // Compute distance between particles

    // Short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

void move(particle_t& p, double size) {
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Boundary conditions
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

double lower_bound;
double upper_bound;
int local_num_parts; // Define a local variable to store the number of particles per rank

int printingRank = 0; // Set this to control which rank prints debug output

std::vector<particle_t> local_parts;

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    double row_height = size / num_procs;
    lower_bound = rank * row_height;
    upper_bound = (rank + 1) * row_height;
    
    // Store local particles relevant to this rank
    local_parts.clear();
    for (int i = 0; i < num_parts; i++) {
        if (parts[i].y >= lower_bound && parts[i].y < upper_bound) {
            local_parts.push_back(parts[i]);
        }
    }
}

int counter = 0;

// void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    
//     // std::cout << "rank [" << rank << "] counter[" << counter << "]\n";
//     // counter ++;
//     MPI_Barrier(MPI_COMM_WORLD);
//     // if (rank == 0)
//     // {
//     //     std::cout << "==================================\n";
//     // }
//     // ============================== SEND / RECEIVE GHOST PARTICLES ================================= //
//     // Vectors to store particles that need to be sent and received
//     std::vector<double> ghost_to_above;
//     std::vector<double> ghost_to_below;
//     std::vector<double> ghost_from_above;
//     std::vector<double> ghost_from_below;

//     // Define rank above and below
//     int rank_above = (rank + 1 < num_procs) ? rank + 1 : -1;
//     int rank_below = (rank - 1 >= 0) ? rank - 1 : -1;

//     // Iterate over local particles to determine which need to be sent
//     for (size_t i = 0; i < local_parts.size(); i++) {
//         if (upper_bound - local_parts[i].y < cutoff) {
//             ghost_to_above.push_back(local_parts[i].x);
//             ghost_to_above.push_back(local_parts[i].y);
//         }
//         if (local_parts[i].y - lower_bound < cutoff) {
//             ghost_to_below.push_back(local_parts[i].x);
//             ghost_to_below.push_back(local_parts[i].y);
//         }
//     }

//     // Send to rank above and receive from rank below
//     if (rank_above != -1) {
//         int ghost_to_above_size = ghost_to_above.size();
//         int ghost_from_below_size = 0;
//         MPI_Sendrecv(&ghost_to_above_size, 1, MPI_INT, rank_above, 0,
//                      &ghost_from_below_size, 1, MPI_INT, rank_below, 0,
//                      MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//         // std::cout << "[Rank " << rank << "] Sent " << ghost_to_above_size / 2 
//         //           << " ghost particles to Rank " << rank_above 
//         //           << " | Received " << ghost_from_below_size / 2 
//         //           << " ghost particles from Rank " << rank_below << std::endl;
//     }

//     // Send to rank below and receive from rank above
//     if (rank_below != -1) {
//         int ghost_to_below_size = ghost_to_below.size();
//         int ghost_from_above_size = 0;
//         MPI_Sendrecv(&ghost_to_below_size, 1, MPI_INT, rank_below, 1,
//                      &ghost_from_above_size, 1, MPI_INT, rank_above, 1,
//                      MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//         // std::cout << "[Rank " << rank << "] Sent " << ghost_to_below_size / 2 
//         //           << " ghost particles to Rank " << rank_below 
//         //           << " | Received " << ghost_from_above_size / 2 
//         //           << " ghost particles from Rank " << rank_above << std::endl;
//     }

//     MPI_Barrier(MPI_COMM_WORLD);

//     // ============================== MOVE PARTICLES ================================= //

//     for (int i = 0; i < local_parts.size(); ++i) {
//         move(local_parts[i], local_parts.size());
//     }

//     MPI_Barrier(MPI_COMM_WORLD);
// }

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    std::cout << "Process " << rank << " reached before the barrier." << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "Process " << rank << " passed the barrier." << std::endl;
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.
    // Print the x coordinate of the first particle
}