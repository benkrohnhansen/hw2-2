#include "common.h"
#include <mpi.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <cstring>
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

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    
    // ============================== SEND / RECEIVE GHOST PARTICLES ================================= //
    // Vectors to store particles that need to be sent and received
    std::vector<double> ghost_to_above;
    std::vector<double> ghost_to_below;
    std::vector<double> ghost_from_above;
    std::vector<double> ghost_from_below;

    // Define rank above and below
    int rank_above = (rank + 1 < num_procs) ? rank + 1 : -1;
    int rank_below = (rank - 1 >= 0) ? rank - 1 : -1;

    // Iterate over local particles to determine which need to be sent
    for (size_t i = 0; i < local_parts.size(); i++) {
        if (upper_bound - local_parts[i].y < cutoff) {
            ghost_to_above.push_back(local_parts[i].x);
            ghost_to_above.push_back(local_parts[i].y);
        }
        if (local_parts[i].y - lower_bound < cutoff) {
            ghost_to_below.push_back(local_parts[i].x);
            ghost_to_below.push_back(local_parts[i].y);
        }
    }

    MPI_Request requests[4];
    int req_count = 0;

    // Send ghost_to_above to rank_above and receive ghost_from_above from rank_above
    if (rank_above != -1) {
        int ghost_to_above_size = ghost_to_above.size();
        MPI_Isend(&ghost_to_above_size, 1, MPI_INT, rank_above, 0, MPI_COMM_WORLD, &requests[req_count++]);
        MPI_Isend(ghost_to_above.data(), ghost_to_above_size, MPI_DOUBLE, rank_above, 1, MPI_COMM_WORLD, &requests[req_count++]);
        
        int ghost_from_above_size;
        MPI_Irecv(&ghost_from_above_size, 1, MPI_INT, rank_above, 2, MPI_COMM_WORLD, &requests[req_count++]);
        MPI_Wait(&requests[req_count - 1], MPI_STATUS_IGNORE);
        ghost_from_above.resize(ghost_from_above_size);
        MPI_Irecv(ghost_from_above.data(), ghost_from_above_size, MPI_DOUBLE, rank_above, 3, MPI_COMM_WORLD, &requests[req_count++]);
    }

    // Send ghost_to_below to rank_below and receive ghost_from_below from rank_below
    if (rank_below != -1) {
        int ghost_to_below_size = ghost_to_below.size();
        MPI_Isend(&ghost_to_below_size, 1, MPI_INT, rank_below, 2, MPI_COMM_WORLD, &requests[req_count++]);
        MPI_Isend(ghost_to_below.data(), ghost_to_below_size, MPI_DOUBLE, rank_below, 3, MPI_COMM_WORLD, &requests[req_count++]);
        
        int ghost_from_below_size;
        MPI_Irecv(&ghost_from_below_size, 1, MPI_INT, rank_below, 0, MPI_COMM_WORLD, &requests[req_count++]);
        MPI_Wait(&requests[req_count - 1], MPI_STATUS_IGNORE);
        ghost_from_below.resize(ghost_from_below_size);
        MPI_Irecv(ghost_from_below.data(), ghost_from_below_size, MPI_DOUBLE, rank_below, 1, MPI_COMM_WORLD, &requests[req_count++]);
    }

    // Wait for all communication to complete
    MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);

    // ============================= Compute Forces ============================= //
    for (int i = 0; i < local_parts.size(); ++i) {
        local_parts[i].ax = local_parts[i].ay = 0;
        for (int j = 0; j < local_parts.size(); ++j) {
            apply_force(local_parts[i], local_parts[j]);
        }
        for (int jj = 0; jj < ghost_from_above.size(); jj+=2) {
            apply_force(local_parts[i], ghost_from_above[jj], ghost_from_above[jj + 1]);
        }
        for (int jjj = 0; jjj < ghost_from_below.size(); jjj+=2) {
            apply_force(local_parts[i], ghost_from_below[jjj], ghost_from_below[jjj + 1]);
        }
    }

    // ============================== MOVE PARTICLES ================================= //
    for (int i = 0; i < local_parts.size(); ++i) {
        move(local_parts[i], local_parts.size());
        if (local_parts[i].y > upper_bound) {
            // Move to rank above
        }
        if (local_parts[i].y < lower_bound) {
            // Move to rank below
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Print first particle's position, velocity, and acceleration in rank 0 for debugging
    if (rank == 0 && !local_parts.empty()) {
        std::cout << "[Rank 0] First particle after move: "
                  << "Position: (" << local_parts[0].x << ", " << local_parts[0].y << ") "
                  << "Velocity: (" << local_parts[0].vx << ", " << local_parts[0].vy << ") "
                  << "Acceleration: (" << local_parts[0].ax << ", " << local_parts[0].ay << ") "
                  << std::endl;
    }
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.
    // Print the x coordinate of the first particle
}