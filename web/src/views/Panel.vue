<template>
  <Forbidden v-if="!store.getters.userIsLoggedIn"/>
  <AdminPanel v-if="store.getters.userRole === 'admin'"/>
  <UserPanel v-else/>
</template>

<script>
import {useStore} from "vuex";
import AdminPanel from "@/components/AdminPanel";
import Forbidden from "@/views/Forbidden";
import UserPanel from "@/components/UserPanel";


export default {
  name: "Panel",
  components: {UserPanel, AdminPanel, Forbidden},
  setup() {
    const store = useStore();

    return {
      store
    }
  }
}


</script>

<style scoped>


</style>