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
from abc import abstractmethod, ABC
from copy import copy
import numpy as np
from sklearn.utils import check_random_state
from subclass_register import SubclassRegister
# import random


class NetworkPiece:
    def __init__(self, start_idx, num_nodes, num_chunks, random_state=None):
        self.start_idx = start_idx
        self.num_nodes = num_nodes
        self.active_nodes = num_nodes

        self.chunks = self.generate_chunks(num_chunks)
        self.rng = check_random_state(random_state)

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
            self.rng.shuffle(idxes)

        nodes_per_chunk = self.num_nodes // num_chunks

        return [set(idxes[i*nodes_per_chunk:(i+1)*nodes_per_chunk]) for i in range(num_chunks)]

    def get_random_chunk_idx(self, n):
        """Get ``n`` random chunk indexes.
        """
        chunk_indexes = list(range(len(self.chunks)))
        return self.rng.choice(chunk_indexes, n, replace=False)
    
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


class EvolvingSubNetworkComponent:
    def __init__(
        self,
        network,
        init_size,
        prob_adding=1,
        prob_removing=1,
        prob_shifting=0,
        prob_becoming_passive=0,
        activation_rate=None,
        random_state=None,
    ):
        """

        network : NetworkPiece
        init_size : int
        prob_adding : float
            Per time-step probability of activating a passive chunk
        prob_removing : float
            Per time-step probability of deactivating an active chunk
        prob_becoming_passive : float
            Per time-step probability of a deactivated chunk becoming passive
        prob_shifting : float
            Per time-step probability of the node indices shifting
        activation_rate : float or None
            How fast a node goes from active to deactivated
        random_state : Int or None or numpy.random.RandomState
        """
        self.network = network

        self.init_size = init_size
        self.prob_adding = prob_adding
        self.prob_removing = prob_removing
        self.prob_becoming_passive = prob_becoming_passive
        self.prob_shifting = prob_shifting
        
        self.activation_rate = activation_rate
        self.random_state = random_state

    def init_subnetwork(
        self,
    ):
        self.chunk_idxes = set(range(len(self.network.chunks)))
        self.active_chunks = set(self.network.get_random_chunk_idx(self.init_size))
        self.deactivated_chunks = set()
        self.chunk_weights = {
            c: 0.9*(c in self.active_chunks) for c in self.chunk_idxes
        }
        self.shift = 0
        self.rng_ = check_random_state(self.random_state)

    def evolve_one_step(self):
        self.shift_phase()
        self.remove_phase()
        self.add_phase()
        self.make_passive_phase()
        self.update_weights()

    def add_chunk(self):
        """Adds a random chunk to the subnetwork.
        """
        available = self.chunk_idxes - self.deactivated_chunks - self.active_chunks
        
        if len(available) < 1:
            return 

        added = self.rng_.choice(list(available))
        self.active_chunks.add(added)
        self.chunk_weights[added] = 0.01

    def remove_chunk(self):
        """Removes a random chunk to the subnetwork.
        """
        if len(self.active_chunks) == 0:
            return

        removed = self.rng_.choice(list(self.active_chunks))
        self.active_chunks.remove(removed)
        self.deactivated_chunks.add(removed)
    
    def make_chunk_passive(self):
        """Removes a random chunk to the subnetwork.
        """
        if len(self.deactivated_chunks) == 0:
            return

        made_passive = self.rng_.choice(list(self.deactivated_chunks))
        self.deactivated_chunks.remove(made_passive)
    
    def shift_phase(self):
        """Checks if a the network should shift
        """
        draw = self.rng_.uniform()
        if draw < self.prob_shifting:
            self.shift += 1

    def remove_phase(self):
        """Checks if a chunk should be removed to the and removes chunk accordingly
        """
        draw = self.rng_.uniform()
        if draw < self.prob_removing:
            self.remove_chunk()

    def add_phase(self):
        """Checks if a chunk should be added to the and adds chunk accordingly
        """
        draw = self.rng_.uniform()
        if draw < self.prob_adding:
            self.add_chunk()

    def make_passive_phase(self):
        """Checks if a chunk should be removed to the and removes chunk accordingly
        """
        draw = self.rng_.uniform()
        if draw < self.prob_becoming_passive:
            self.make_chunk_passive()

    def compute_next_weight(self, chunk_weight, active):
        if self.activation_rate is None:
            if active:
                return 0.99
            else:
                return 0.01

        if active:
            activation_rate = self.activation_rate + 1
        else:
            activation_rate = 1 - self.activation_rate

        K = activation_rate/self.activation_rate
        weight = activation_rate*chunk_weight*(1 - chunk_weight/K)
        
        if weight < 0.01:
            weight = 0.01
        elif weight > 0.99:
            weight = 0.99
        
        return weight

    def update_weights(self):
        for chunk_idx in self.chunk_idxes:
            active = chunk_idx in self.active_chunks
            self.chunk_weights[chunk_idx] = self.compute_next_weight(self.chunk_weights[chunk_idx], active)

    def get_factor_column(self, debug=False):
        """Generate a single random static tensor factor based on the subnetworks inner state.
        """
        min_idx = self.network.nodes_in_chunks[0]
        factor_column = self.rng_.standard_normal((self.network.num_nodes))*0.1 + 0.2

        for chunk_idx, chunk_weight in self.chunk_weights.items():
            chunk = self.network.chunks[chunk_idx]
            for idx in chunk:
                idx = min_idx + ((idx - min_idx + self.shift) % self.network.active_nodes)
                factor_column[idx] += (self.rng_.standard_normal()*chunk_weight*0.1) + chunk_weight*0.8

        return factor_column
    
    def get_dynamic_factor(self, num_timesteps, debug=False):
        """Generate the full dynamic tensor factor.
        """
        self.init_subnetwork()
        dynamic_factor = []
        for _ in range(num_timesteps):
            self.evolve_one_step()
            dynamic_factor.append(self.get_factor_column(debug=debug))
        
        return np.vstack(dynamic_factor)


evolving_component_register = SubclassRegister('evolving component generator')


@evolving_component_register.link_base
class BaseEvolvingComponentsGenerator(ABC):
    @abstractmethod
    def init_components(self):
        raise NotImplementedError

    @abstractmethod
    def generate_factors(self):
        raise NotImplementedError


class EvolvingNetworksGenerator(BaseEvolvingComponentsGenerator):
    def __init__(
            self,
            component_params,
            num_timesteps,
            num_nodes,
            num_chunks=None,
            chunk_size=None,
            non_overlapping=True,
            random_state=None,
            **kwargs
        ):
        """
        component_params = [
            {
                "init_size": 10,
                "prob_adding": 0.4,
                "prob_removing": 0.6,
                "prob_shifting": 0,
                "prob_becoming_passive": 0.01,
                "activation_rate": None,
            },
            ...
        ]
        """
        self.num_timesteps = num_timesteps
        self.num_components = len(component_params)
        self.component_params = component_params
        self.random_state = random_state

        if num_chunks is None and chunk_size is None:
            num_chunks = num_nodes
        elif num_chunks is None and chunk_size is not None:
            num_chunks = num_nodes//chunk_size
        elif num_chunks is not None and chunk_size is not None:
            raise ValueError("Cannot specify both no. chunks and chunk size.")

        self.network = Network(num_nodes=num_nodes, num_chunks=num_chunks)

        if non_overlapping:
            self.subnetworks = self.network.generate_pieces(self.num_components)
        else:
            self.subnetworks = [self.network]*self.num_components

    def init_components(self):
        rng = check_random_state(self.random_state)
        self.generators = [
            EvolvingSubNetworkComponent(subnetwork, random_state=rng, **kwargs) 
            for subnetwork, kwargs in zip(self.subnetworks, self.component_params)
        ]


    def generate_factors(self):
        self.init_components()
        components = [
            generator.get_dynamic_factor(self.num_timesteps)
            for generator in self.generators
        ]
        return np.array(components).transpose(1, 2, 0)


class RandomNetworkGenerator(BaseEvolvingComponentsGenerator):
    def __init__(
        self,
        num_timesteps,
        num_nodes,
        num_components,
        mean=0,
        std=0.1,
        use_parafac2=True,
        phi_off_diags=None,
        random_state=None,
    ):
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
        self.random_state = random_state

    def init_components(self):
        pass

    def generate_factor_blueprint(self, random_state):
        rng = check_random_state(random_state)

        if self.phi_off_diags is not None:
            phi = np.ones(shape=(self.num_components, self.num_components))*self.phi_off_diags
            np.fill_diagonal(phi, val=1)
            return np.linalg.cholesky(phi)
        return rng.standard_normal((self.num_components, self.num_components))*self.std + self.mean

    def generate_parafac2_network(self, random_state):
        rng = check_random_state(random_state)

        factor_blueprint = self.generate_factor_blueprint(rng)
        factors = []
        for t in range(self.num_timesteps):
            rand_orth = np.linalg.qr(rng.randn(self.num_nodes, self.num_components))[0]
            factors.append(rand_orth@factor_blueprint)
        return np.array(factors).transpose(0, 1, 2)

    def generate_factors(self):
        rng = check_random_state(self.random_state)

        if self.use_parafac2:
            return self.generate_parafac2_network(rng)
        else:
            return self.mean + self.std*rng.standard_normal(
                (self.num_nodes, self.num_components, self.num_timesteps)
            )


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