import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import store from './store';

const savedState = JSON.parse(localStorage.getItem('vuex_state')) || {};

store.replaceState(savedState)

createApp(App).use(router).use(store).mount('#app')
