"""
Network-structure tensor components

---------------
Network classes
---------------

The purpose of the network classes is to provide node indices and
node chunks to the subnetwork classes, which again generate the
evolving tensor components.

A network consists of a set of chunks, which are goups of nodes
that have correlated behaviour.

A network is a network piece whose indices start at zero.

A network can generate network pieces (which are again networks).
 - This is done to create non-overlapping subnetworks

------------------
Subnetwork classes
------------------

A subnetwork has a network piece which it adopts node-indexes from.

A subnetwork can generate tensor components drawn from a random 
distribution parametrised by its its inner state.

A subnetwork can evolve its inner state by single timesteps.
"""
from copy import copy
import numpy as np
# import random


class NetworkPiece:
    # TODO: kanskje Network heller skal arve denne?
    def __init__(self, start_idx, num_nodes, num_chunks):
        self.start_idx = start_idx
        self.num_nodes = num_nodes
        self.active_nodes = num_nodes

        self.chunks = self.generate_chunks(num_chunks)

    @property
    def num_chunks(self):
        return len(self.chunks)
    
    @property
    def nodes_in_chunks(self):
        """Return a list of the nodes contained in a chunk.
        """
        return sorted(list(set().union(*self.chunks)))

    def generate_idxes(self):
        return np.arange(self.start_idx, self.start_idx+self.num_nodes)

    def generate_chunks(self, num_chunks, shuffle=False):
        """Separate the node-indices into the correct chunks.
        """
        idxes = self.generate_idxes()

        if shuffle:
            np.random.shuffle(idxes)

        nodes_per_chunk = self.num_nodes // num_chunks

        return [set(idxes[i*nodes_per_chunk:(i+1)*nodes_per_chunk]) for i in range(num_chunks)]

    def get_random_chunk_idx(self, n):
        """Get ``n`` random chunk indexes.
        """
        chunk_indexes = list(range(len(self.chunks)))
        return np.random.choice(chunk_indexes, n, replace=False)
    
    def generate_pieces(self, num_pieces):
        """Separate the network into non-overlapping network pieces.

        Each piece corresponds to one tensor component.
        """
        network_pieces = []
        chunks_per_piece = self.num_chunks//num_pieces

        for i in range(num_pieces):
            network_piece = copy(self)
            network_piece.chunks = network_piece.chunks[i*chunks_per_piece:(i+1)*chunks_per_piece]
            nodes_per_chunk = [len(chunk) for chunk in network_piece.chunks]
            network_piece.active_nodes = sum(nodes_per_chunk)
            network_pieces.append(network_piece)
        return network_pieces


class Network(NetworkPiece):
    def __init__(self, num_nodes, num_chunks):
        """Initiate a chunked network with non-overlapping chunks.
        """
        super().__init__(0, num_nodes, num_chunks)


class SubNetwork:
    def __init__(self, network):
        self.network = network

    def init_subnetwork(self, init_size, prob_adding=1, prob_removing=1):
        """Set subnetwork parameters and set its inner state.
        """
        self.chunk_idxs = self.network.get_random_chunk_idx(init_size)
        chunk_weight = 1
        self.chunk_idxs = {c: chunk_weight for c in self.chunk_idxs}
        self.previously_removed = []

        self.prob_adding = prob_adding
        self.prob_removing = prob_removing
        self.prob_readding = prob_removing

    def add_chunk(self):
        """Activates a random chunk.
        """
        # TODO: skal man kunne ha en liten sjanse for å legge til prev removed?
        chunk_idxs = set(range(self.network.num_chunks))
        removed_idxs = set(self.previously_removed)
        curr_idxs = set(self.chunk_idxs)
        available = chunk_idxs - removed_idxs - curr_idxs
        
        #print(available)
        if len(available)<1:
            return 

        added = np.random.choice(list(available))
        self.chunk_idxs[added] = 1

    def remove_chunk(self):
        """Deactivates a random chunk in the subnetwork.
        """
        if len(self.chunk_idxs) > 0:
            removed = np.random.choice(list(self.chunk_idxs.keys()))
            self.chunk_idxs[removed] = 0
            self.previously_removed.append(removed)

    def add_phase(self):
        """Checks if a chunk should be added to the and adds chunk accordingly
        """
        draw = np.random.uniform()
        if draw < self.prob_adding:
            self.add_chunk()

    def remove_phase(self):
        """Checks if a chunk should be removed to the and removes chunk accordingly
        """
        draw = np.random.uniform()
        if draw < self.prob_removing:
            self.remove_chunk()

    def evolve_one_step(self):
        """Evolve the sub-network one step.
        """
        self.remove_phase()
        self.add_phase()
  
    def get_factor_column(self, debug=False):
        """Generate a single random static tensor factor based on the subnetworks inner state.
        """
        factor_column = np.random.standard_normal((self.network.num_nodes))*0.1 + 0.2

        for chunk_idx, chunk_strength in self.chunk_idxs.items():
            chunk = self.network.chunks[chunk_idx]
            for idx in chunk:
                factor_column[idx] += (np.random.standard_normal()*chunk_strength*0.1) + chunk_strength*0.8

        return factor_column


    @property
    def chunks(self):
        return [self.network.chunks[idx] for idx in self.chunk_idxs]


class SmoothSubNetwork(SubNetwork):
    # This was added to test some other functionality and is not really used.
    #TODO: istedet for å ha chunk_idxs må vi ha to dicts en for om den er aktiv og en for styrke
    #TODO: legg til en weight update fase

    def init_subnetwork(self, init_size, prob_adding=1, prob_removing=1, activation_rate=0.1):
        super().init_subnetwork(
            init_size=init_size,
            prob_adding=prob_adding,
            prob_removing=prob_removing
        )
        self.activation_rate = activation_rate
        self.chunk_idxs = {
            c: 0.9 for c in self.chunk_idxs
        }
        self.chunk_states = {
            c: True for c in self.chunk_idxs
        }

    def evolve_one_step(self):
        self.remove_phase()
        self.add_phase()
        self.update_weights()

    def add_chunk(self):
        """Adds a random chunk to the subnetwork.
        """
        # TODO: skal man kunne ha en liten sjanse for å legge til prev removed?
        chunk_idxs = set(range(self.network.num_chunks))
        removed_idxs = set(self.previously_removed)
        curr_idxs = set(self.chunk_idxs)
        available = chunk_idxs - removed_idxs - curr_idxs
        
        #print(available)
        if len(available)<1:
            return 

        added = np.random.choice(list(available))
        self.chunk_states[added] = True
        self.chunk_idxs[added] = 0.01

    def remove_chunk(self):
        """Removes a random chunk to the subnetwork.
        """
        if len(self.chunk_idxs) > 0:
            removed = np.random.choice(list(self.chunk_idxs.keys()))
            self.chunk_states[removed] = False
            self.previously_removed.append(removed)

    def update_strength(self, chunk_strength, active):
        if active:
            activation_rate = self.activation_rate + 1

        else:
            activation_rate = 1 - self.activation_rate

        K = activation_rate/self.activation_rate
        strength = activation_rate*chunk_strength*(1 - chunk_strength/K)
        
        if strength < 0.01:
            strength = 0.01
        elif strength > 0.99:
            strength = 0.99
        
        return strength

    def update_weights(self):
        for chunk_idx, chunk_strength in self.chunk_idxs.items():
            active = self.chunk_states[chunk_idx]
            self.chunk_idxs[chunk_idx] = self.update_strength(chunk_strength, active)
            #self.chunk_weight[node] = (self.r+1)*self.chunk_weight[node] - self.r*self.chunk_weight[node]**2/self.K


class ShiftedSubNetwork(SubNetwork):
    """A subnetwork that has a probability of shifting its indices.
    """
    def init_subnetwork(self, init_size, prob_shifting=0.5, prob_adding=0, prob_removing=0):
        self.prob_shifting = prob_shifting
        self.shift = 0
        super().init_subnetwork(init_size=init_size, prob_adding=prob_adding, prob_removing=prob_removing)
    
    def shift_phase(self):
        """Checks if a the network should shift
        """
        draw = np.random.uniform()
        if draw < self.prob_shifting:
            self.shift += 1
    
    def evolve_one_step(self):
        self.shift_phase()
        super().evolve_one_step()
        
    def get_factor_column(self, debug=False):
        min_idx = self.network.nodes_in_chunks[0]
        max_idx = self.network.nodes_in_chunks[-1]

        factor_column = np.random.standard_normal((self.network.num_nodes))*0.1 + 0.2
        if debug:
            factor_column = np.zeros((self.network.num_nodes))

        for chunk_idx, chunk_strength in self.chunk_idxs.items():
            chunk = self.network.chunks[chunk_idx]
            for idx in chunk:

                idx = min_idx + ((idx - min_idx + self.shift) % self.network.active_nodes)
                factor_column[idx] += (np.random.standard_normal()*chunk_strength*0.1) + chunk_strength*0.8
                if debug:
                    factor_column[idx] = chunk_strength
        return factor_column


class NetworkFactorGenerator:
    def __init__(self, network_params, num_timesteps, generators, init_kwargs):
        """
        network_params : {'network_type': Network, 'network_kwargs': {<KWARGS>}}
        """

        self.network = network_params['network_type'](**network_params['network_kwargs'])

        self.generators = [Generator(self.network,**kwargs) for Generator, kwargs in generators]
        self.init_kwargs = init_kwargs
        self.num_timesteps = num_timesteps


    def generate_factors(self):
        for generator, init_kwarg in zip(self.generators, self.init_kwargs):
            generator.init_subnetwork(**init_kwarg)
        
        components = []
        for generator in self.generators:
            component = []
            for i in range(self.num_timesteps):
                generator.evolve_one_step()
                component.append(generator.get_factor_column())
            components.append(component)
        return np.array(components).transpose(2,0,1)


class NonOverlappingNetworkFactorGenerator(NetworkFactorGenerator):
    def __init__(self, network_params, num_timesteps, generators, init_kwargs):
        assert len(generators) == len(init_kwargs), "Need the same number of generators as init keyword arguments"

        self.num_components = len(generators)

        # Generate overarching network and non-overlapping network pieces
        self.network = network_params['network_type'](**network_params['network_kwargs'])
        self.networks = self.network.generate_pieces(num_pieces=self.num_components)

        # Create the subnetworks
        self.generators = [
            Generator(network_pice, **kwargs) for network_pice, (Generator, kwargs) in zip(self.networks, generators)
        ]
        self.init_kwargs = init_kwargs
        self.num_timesteps = num_timesteps


class RandomNetworkFactorGenerator:
    def __init__(self, num_timesteps, num_nodes, num_components, mean=0, std=0.1, use_parafac2=True, phi_off_diags=None):
        """
        network_params : {'network_type': Network, 'network_kwargs': {<KWARGS>}}
        """
        self.mean = mean
        self.std = std
        self.num_components = num_components
        self.num_timesteps = num_timesteps
        self.num_nodes = num_nodes
        self.use_parafac2 = use_parafac2
        self.phi_off_diags = phi_off_diags

    def generate_factor_blueprint(self):
        if self.phi_off_diags is not None:
            phi = np.ones(shape=(self.num_components, self.num_components))*self.phi_off_diags
            np.fill_diagonal(phi, val=1)
            return np.linalg.cholesky(phi)
        return np.random.randn(self.num_components, self.num_components)*self.std + self.mean

    def generate_parafac2_network(self):
        factor_blueprint = self.generate_factor_blueprint()
        factors = []
        for t in range(self.num_timesteps):
            rand_orth = np.linalg.qr(np.random.randn(self.num_nodes, self.num_components))[0]
            factors.append(rand_orth@factor_blueprint)
        return np.array(factors).transpose(0, 1, 2)

    def generate_factors(self):
        if self.use_parafac2:
            return self.generate_parafac2_network()
        else:
            return np.random.randn(self.num_nodes, self.num_components, self.num_timesteps)*self.std + self.mean


if __name__ == "__main__":
    np.random.seed(0)
    #random.seed(0)
    g = Network(10, 10)

    sb = ShiftedSubNetwork(g)
    sb.init_subnetwork(3)

    # print(sb.chunks)

    # print(sb.get_factor_column().T)
    
    for _ in range(3):
        sb.evolve_one_step()

        #print(sb.chunks)
        print(sb.get_factor_column(debug=False).T)