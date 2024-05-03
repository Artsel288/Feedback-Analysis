export const localStoragePlugin = store => {
    // This will be called whenever the state changes
    store.subscribe((mutation, state) => {
        // Save the state to localStorage
        localStorage.setItem('vuex_state', JSON.stringify(state));
    });
};
