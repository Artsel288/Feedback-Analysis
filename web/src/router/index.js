import {createRouter, createWebHistory} from 'vue-router'
import Profile from '@/views/Profile.vue';
import Login from '@/views/Login.vue';
import Register from '@/views/Register.vue';
import Panel from "@/views/Panel";
import EditWebinar from "@/views/EditWebinar";
import Webinar from "@/views/Webinar";

const routes = [
    {path: '/', component: Profile},
    {path: '/login', component: Login},
    {path: '/register', component: Register},
    {path: '/panel', component: Panel},
    {path: '/edit-webinar/:id', component: EditWebinar},
    {path: '/webinar/:id', component: Webinar},
]

const router = createRouter({
    history: createWebHistory(process.env.BASE_URL),
    routes
})

export default router
