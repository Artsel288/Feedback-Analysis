import { createStore } from 'vuex';
import { localStoragePlugin } from '@/store/localStoragePlugin';

const store = createStore({
    state: {
        userIsLoggedIn: false,
        userRole: '', // Possible values: 'admin', 'teacher'
    },
    mutations: {
        setUserLoggedIn(state, status) {
            state.userIsLoggedIn = status;
        },
        setUserRole(state, role) {
            state.userRole = role;
        }
    },
    actions: {
        setUserLoggedIn({ commit }, status) {
            commit('setUserLoggedIn', status);
        },
        setUserRole({ commit }, role) {
            commit('setUserRole', role);
        }
    },
    getters: {
        userIsLoggedIn: state => state.userIsLoggedIn,
        userRole: state => state.userRole
    },
    plugins: [localStoragePlugin]
});

export default store;