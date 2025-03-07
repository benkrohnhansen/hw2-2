#include "common.h"
#include <mpi.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <cstring>
#include <algorithm>
// Put any static global variables here that you will use throughout the simulation.

const std::vector<std::pair<int, int>> neighboring_cell = {
    {-1, -1}, {0, -1}, {1, -1},
    {-1, 0}, {0, 0}, {1, 0},
    {-1, 1}, {0, 1}, {1, 1}
};

std::vector<std::vector<particle_t>> tiles;
std::vector<std::vector<particle_t>> ghost_from_above_tiles;
std::vector<std::vector<particle_t>> ghost_from_below_tiles;


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

std::vector<particle_t> local_parts;

int num_tiles_x;
int num_tiles_y;

double len_tiles_x;
double len_tiles_y;

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

    num_tiles_x = std::floor(size / cutoff);
    num_tiles_y = std::floor(row_height/ cutoff);
    
    len_tiles_x = size / num_tiles_x;
    len_tiles_y = row_height/ num_tiles_y;

    tiles.resize(num_tiles_x * num_tiles_y);
    ghost_from_above_tiles.resize(num_tiles_x);
    ghost_from_below_tiles.resize(num_tiles_x);

    MPI_Barrier(MPI_COMM_WORLD);
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // ============================== MOVE PARTICLES ================================= //
    // std::vector<double> ghost_to_above;
    // std::vector<double> ghost_to_below;
    // std::vector<double> ghost_from_above;
    // std::vector<double> ghost_from_below;

    std::vector<particle_t> ghost_to_above;
    std::vector<particle_t> ghost_to_below;
    std::vector<particle_t> ghost_from_above;
    std::vector<particle_t> ghost_from_below;

    // Define rank above and below
    int rank_above = (rank + 1 < num_procs) ? rank + 1 : -1;
    int rank_below = (rank - 1 >= 0) ? rank - 1 : -1;

    // Iterate over local particles to determine ghost particle counts
    // for (size_t i = 0; i < local_parts.size(); i++) {
    //     if (upper_bound - local_parts[i].y < cutoff) {
    //         ghost_to_above.push_back(local_parts[i].x);
    //         ghost_to_above.push_back(local_parts[i].y);
    //     }
    //     if (local_parts[i].y - lower_bound < cutoff) {
    //         ghost_to_below.push_back(local_parts[i].x);
    //         ghost_to_below.push_back(local_parts[i].y);
    //     }
    // }

    for (size_t i = 0; i < local_parts.size(); i++) {
        if (upper_bound - local_parts[i].y < cutoff) {
            ghost_to_above.push_back(local_parts[i]);
        }
        if (local_parts[i].y - lower_bound < cutoff) {
            ghost_to_below.push_back(local_parts[i]);
        }
    }

    // Set ghost counts based on vector sizes
    int ghost_to_above_count = ghost_to_above.size();
    int ghost_to_below_count = ghost_to_below.size();
    int ghost_from_above_count = 0;
    int ghost_from_below_count = 0;

    // ============================== SEND / RECEIVE GHOST PARTICLE COUNTS ================================= //
    MPI_Request requests[4]; // Array for non-blocking communication

    // Non-blocking send and receive for ghost counts
    MPI_Isend(&ghost_to_above_count, 1, MPI_INT, rank_above, 0, MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(&ghost_from_below_count, 1, MPI_INT, rank_below, 0, MPI_COMM_WORLD, &requests[1]);
    MPI_Isend(&ghost_to_below_count, 1, MPI_INT, rank_below, 1, MPI_COMM_WORLD, &requests[2]);
    MPI_Irecv(&ghost_from_above_count, 1, MPI_INT, rank_above, 1, MPI_COMM_WORLD, &requests[3]);

    // Wait for all sends/receives to complete
    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);

    // Resize vectors based on received counts
    ghost_from_above.resize(ghost_from_above_count);
    ghost_from_below.resize(ghost_from_below_count);

    // ============================== SEND / RECEIVE GHOST PARTICLE DATA ================================= //
    if (ghost_to_above_count > 0 && rank_above >= 0) {
        MPI_Isend(ghost_to_above.data(), ghost_to_above_count, PARTICLE, rank_above, 2, MPI_COMM_WORLD, &requests[0]);
    }
    if (ghost_from_below_count > 0 && rank_below >= 0) {
        MPI_Irecv(ghost_from_below.data(), ghost_from_below_count, PARTICLE, rank_below, 2, MPI_COMM_WORLD, &requests[1]);
    }

    if (ghost_to_below_count > 0 && rank_below >= 0) {
        MPI_Isend(ghost_to_below.data(), ghost_to_below_count, PARTICLE, rank_below, 3, MPI_COMM_WORLD, &requests[2]);
    }
    if (ghost_from_above_count > 0 && rank_above >= 0) {
        MPI_Irecv(ghost_from_above.data(), ghost_from_above_count, PARTICLE, rank_above, 3, MPI_COMM_WORLD, &requests[3]);
    }

    // Wait for all sends/receives to complete
    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
    
    // =================== COMPUTE TILES ======================= //
    // std::vector<std::vector<std::vector<<&particle_t>>>> tiled_local_parts;
    // std::vector<std::vector<double>> tiled_ghost_above;
    // std::vector<std::vector<double>> tiled_ghost_below;
    
    // for (int i = 0; i < local_parts.size(); ++i) {
        //     int x_tile = std::floor(local_parts[i].x / len_tiles_x);
        //     int y_tile = std::floor((local_parts[i].y + lower_bound) / len_tiles_y);
        //     tiled_local_parts[x_tile + y_tile * num_tiles_x].push_back(&local_parts[i]);
        // }
        
        // for (int i = 0; i < ghost_from_above_count; i += 2) {
            //     int x_tile = std::floor(local_parts[i] / len_tiles_x);;
            //     tiled_ghost_above[x_tile].push_back(ghost_from_above[i]);
            //     tiled_ghost_above[x_tile].push_back(ghost_from_above[i + 1])
            // }
            
            // for (int i = 0; i < ghost_from_below_count; i += 2) {
                //     int x_tile = std::floor(local_parts[i] / len_tiles_x);;
                //     tiled_ghost_below[x_tile].push_back(ghost_from_below[i]);
                //     tiled_ghost_below[x_tile].push_back(ghost_from_below[i + 1])
                // }
                
                // // ============================= Compute Forces ============================= //
                // for (int i =  0; i < tiled_local_parts.size(); i++) {
                    //     int up_left = i - 1 + len_tiles_x; int up = i + len_tiles_x; int up_right = i + 1 + len_tiles_x;
                    //     int left    = i - 1;                                         int right    = i + 1;
                    //     int dw_left = i - 1 - len_tiles_x; int dw = i - len_tiles_x; int dw_right = i + 1 - len_tiles_x;
                    //     if (i - len_tiles_x < 0) 
                    // }
                    
                    // // ============================= Compute Forces ============================= //
                    // for (int i = 0; i < local_parts.size(); ++i) {
                        //     local_parts[i].ax = local_parts[i].ay = 0;
                        //     for (int j = 0; j < local_parts.size(); ++j) {
                            //         apply_force(local_parts[i], local_parts[j]);
                            //     }
                            //     // Compute forces with ghost particles from above
                            //     for (size_t jj = 0; jj < ghost_from_above.size(); jj += 2) {
                                //         apply_force(local_parts[i], ghost_from_above[jj], ghost_from_above[jj + 1]);
                                //     }
                                
                                //     // Compute forces with ghost particles from below
                                //     for (size_t jj = 0; jj < ghost_from_below.size(); jj += 2) {
                                    //         apply_force(local_parts[i], ghost_from_below[jj], ghost_from_below[jj + 1]);
                                    //     }
                                    // }

    for (auto& cell : tiles) {
        cell.clear();
    }

    for (auto& cell : ghost_from_above_tiles) {
        cell.clear();
    }

    for (auto& cell : ghost_from_below_tiles) {
        cell.clear();
    }
    
    for (int i = 0; i < local_parts.size(); i++) {
        int x = static_cast<int>(local_parts[i].x / len_tiles_x);
        int y = static_cast<int>((local_parts[i].y - lower_bound) / len_tiles_y);
        tiles[x + y * num_tiles_x].push_back(local_parts[i]);
        // std::cout << "In local parts" << std::endl;
    }

    for (int i = 0; i < ghost_from_above.size(); i++) {
        int x = static_cast<int>(ghost_from_above[i].x / len_tiles_x);
        ghost_from_above_tiles[x].push_back(ghost_from_above[i]);
        // std::cout << "In ghost above" << std::endl;
    }
    
    for (int i = 0; i < ghost_from_below.size(); i++) {
        int x = static_cast<int>(ghost_from_below[i].x / len_tiles_x);
        ghost_from_below_tiles[x].push_back(ghost_from_below[i]);
        // std::cout << "In ghost below" << std::endl;
    }

    // Iterate through each particle
    for (int i = 0; i < local_parts.size(); i++) {
        local_parts[i].ax = local_parts[i].ay = 0;
        int x = static_cast<int>(local_parts[i].x / len_tiles_x);
        int y = static_cast<int>((local_parts[i].y - lower_bound) / len_tiles_y);

        // Iterate through neighbors (including itself)
        for (auto& cell : neighboring_cell) {
            int neighbor_x = x + cell.first;
            int neighbor_y = y + cell.second;
            if (neighbor_x >= 0 && neighbor_x < num_tiles_x && neighbor_y >= 0 && neighbor_y < num_tiles_y) {
                std::vector<particle_t>& neighbors = tiles[neighbor_x + neighbor_y * num_tiles_x];
                for (particle_t& neighbor : neighbors) {
                    apply_force(local_parts[i], neighbor);
                }
            }
            if (neighbor_x >= 0 && neighbor_x < num_tiles_x && neighbor_y == num_tiles_y) { // Above
                std::vector<particle_t>& neighbors = ghost_from_above_tiles[neighbor_x];
                for (particle_t& neighbor : neighbors) {
                    apply_force(local_parts[i], neighbor);
                }
            }
            if (neighbor_x >= 0 && neighbor_x < num_tiles_x && neighbor_y < 0) { // Below
                std::vector<particle_t>& neighbors = ghost_from_below_tiles[neighbor_x];
                for (particle_t& neighbor : neighbors) {
                    apply_force(local_parts[i], neighbor);
                }
            }
        }
    }

    // ============================== MOVE PARTICLES ============================== //
    for (size_t i = 0; i < local_parts.size(); i++) {
        move(local_parts[i], size);
    }
    // ============================== PARTICLE EXCHANGE ACROSS RANKS ================================= //

    // Vectors to store particles that need to be sent
    std::vector<particle_t> particles_to_above;
    std::vector<particle_t> particles_to_below;

    // Iterate over local particles to identify which need to be sent
    for (size_t i = 0; i < local_parts.size(); ) { // Using i without increment to erase dynamically
        if (local_parts[i].y >= upper_bound) {
            particles_to_above.push_back(local_parts[i]);
            local_parts.erase(local_parts.begin() + i); // Remove from local
        } else if (local_parts[i].y <= lower_bound) {
            particles_to_below.push_back(local_parts[i]);
            local_parts.erase(local_parts.begin() + i); // Remove from local
        } else {
            ++i; // Only increment if no erase happens
        }
    }

    // Count particles to be sent
    int num_particles_to_above = particles_to_above.size();
    int num_particles_to_below = particles_to_below.size();
    int num_particles_from_above = 0;
    int num_particles_from_below = 0;

    // MPI Requests (renamed from `requests` to `particle_requests`)
    MPI_Request particle_requests[4];
    int request_count = 0;

    // ============================== SEND / RECEIVE PARTICLE COUNTS ================================= //

    // Send/Receive counts to/from adjacent ranks
    if (rank_above >= 0) {
        MPI_Isend(&num_particles_to_above, 1, MPI_INT, rank_above, 4, MPI_COMM_WORLD, &particle_requests[request_count++]);
        MPI_Irecv(&num_particles_from_above, 1, MPI_INT, rank_above, 5, MPI_COMM_WORLD, &particle_requests[request_count++]);
    }
    if (rank_below >= 0) {
        MPI_Isend(&num_particles_to_below, 1, MPI_INT, rank_below, 5, MPI_COMM_WORLD, &particle_requests[request_count++]);
        MPI_Irecv(&num_particles_from_below, 1, MPI_INT, rank_below, 4, MPI_COMM_WORLD, &particle_requests[request_count++]);
    }

    // Wait for the counts to be exchanged before proceeding
    MPI_Waitall(request_count, particle_requests, MPI_STATUSES_IGNORE);

    // Resize vectors for receiving particles
    std::vector<particle_t> particles_from_above(num_particles_from_above);
    std::vector<particle_t> particles_from_below(num_particles_from_below);

    // ============================== SEND / RECEIVE PARTICLE DATA ================================= //
    // Reset request count
    request_count = 0;

    // Send and receive actual particle data
    if (num_particles_to_above > 0 && rank_above >= 0) {
        MPI_Isend(particles_to_above.data(), num_particles_to_above, PARTICLE, rank_above, 6, MPI_COMM_WORLD, &particle_requests[request_count++]);
    }
    if (num_particles_from_below > 0 && rank_below >= 0) {
        MPI_Irecv(particles_from_below.data(), num_particles_from_below, PARTICLE, rank_below, 6, MPI_COMM_WORLD, &particle_requests[request_count++]);
    }

    if (num_particles_to_below > 0 && rank_below >= 0) {
        MPI_Isend(particles_to_below.data(), num_particles_to_below, PARTICLE, rank_below, 7, MPI_COMM_WORLD, &particle_requests[request_count++]);
    }
    if (num_particles_from_above > 0 && rank_above >= 0) {
        MPI_Irecv(particles_from_above.data(), num_particles_from_above, PARTICLE, rank_above, 7, MPI_COMM_WORLD, &particle_requests[request_count++]);
    }

    // Wait for all particle transfers to complete
    MPI_Waitall(request_count, particle_requests, MPI_STATUSES_IGNORE);

    // ============================== INSERT RECEIVED PARTICLES ================================= //
    local_parts.insert(local_parts.end(), particles_from_above.begin(), particles_from_above.end());
    local_parts.insert(local_parts.end(), particles_from_below.begin(), particles_from_below.end());

        MPI_Barrier(MPI_COMM_WORLD);
}


void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Get the local number of particles on each rank
    int local_count = local_parts.size();  // Use local_parts.size() instead of num_parts
    std::vector<int> all_counts(num_procs);

    // Gather particle counts from all ranks
    MPI_Gather(&local_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute displacement array for MPI_Gatherv
    std::vector<int> displacements(num_procs, 0);
    if (rank == 0) {
        int offset = 0;
        for (int i = 0; i < num_procs; i++) {
            displacements[i] = offset;
            offset += all_counts[i];
        }
    }

    // Rank 0 already has enough space allocated in `parts`, so no need to resize
    MPI_Gatherv(local_parts.data(), local_count, PARTICLE,
                parts, all_counts.data(), displacements.data(),
                PARTICLE, 0, MPI_COMM_WORLD);

    // Rank 0 sorts the gathered particles by ID
    if (rank == 0) {
        std::sort(parts, parts + num_parts, [](const particle_t& a, const particle_t& b) {
            return a.id < b.id;
        });

        // Print the x-coordinate of the first particle
        if (num_parts > 0) {
            std::cout << "x-coordinate of first particle: "
                      << parts[0].x << std::endl;
        }
    }
}
