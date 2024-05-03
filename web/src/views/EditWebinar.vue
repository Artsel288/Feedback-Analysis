<template>
  <main v-if="!isForbidden" class="mx-auto container">
    <div class="d-flex justify-content-around">
      <div class="teachers">

        <div class="card w-30 mt-3">
          <div class="card-body">
            <h5 class="card-title">Current teacher</h5>
            <div v-if="webinar.teacher_id !== null">
              <p class="card-text">Name: {{webinar.teacher.firstname}} </p>
              <p class="card-text">Surname: {{webinar.teacher.lastname}}</p>
            </div>
            <div v-else>
              <p class="card-text">Teacher not specified </p>
            </div>
            <div class="d-flex">
              <button v-if="webinar.teacher_id !== null" @click="deleteTeacher" type="button" class="btn btn-danger me-2">Delete</button>
            </div>
          </div>
        </div>

        <form class="search d-flex" @submit.prevent="searchTeachers">
          <input class="form-control mt-2" name="search" placeholder="Type to search...">
          <button class="btn btn-primary ms-2 mt-2" type="submit"> Search</button>
        </form>

        <div v-for="teacher in teachers" :key="teacher.id" class="card w-30 mt-3">
          <div class="card-body">
            <p class="card-text">Name: {{teacher.firstname}} </p>
            <p class="card-text">Surname: {{teacher.lastname}}</p>
            <div class="d-flex">
              <button @click="setTeacher(teacher.id)" type="button" class="btn btn-success me-2 w-25">Set</button>
            </div>
          </div>
        </div>

      </div>
      <div class="methodists">

        <div class="card w-30 mt-3">
          <div class="card-body">
            <h5 class="card-title">Current methodist</h5>
            <div v-if="webinar.methodist_id !== null">
              <p class="card-text">Name: {{webinar.methodist.firstname}}</p>
              <p class="card-text">Surname: {{webinar.methodist.lastname}}</p>
            </div>
            <div v-else>
              <p class="card-text">Methodist not specified </p>
            </div>
            <div class="d-flex">
              <button v-if="webinar.methodist_id !== null" @click="deleteMethodist" type="button" class="btn btn-danger me-2">Delete</button>
            </div>
          </div>
        </div>

        <form class="search d-flex" @submit.prevent="searchMethodist">
          <input class="form-control mt-2" name="search" placeholder="Type to search...">
          <button class="btn btn-primary ms-2 mt-2" type="submit"> Search</button>
        </form>

        <div v-for="methodist in methodists" :key="methodist.id" class="card w-30 mt-3">
          <div class="card-body">
            <p class="card-text">Name: {{methodist.firstname}} </p>
            <p class="card-text">Surname: {{methodist.lastname}}</p>
            <div class="d-flex">
              <button @click="setMethodist(methodist.id)" type="button" class="btn btn-success me-2 w-25">Set</button>
            </div>
          </div>
        </div>

      </div>
    </div>
  </main>
  <Forbidden v-else />
</template>

<script>

import {useRouter} from "vue-router";
import {ref, onBeforeMount} from "vue";
import {useStore} from "vuex";
import Forbidden from "@/views/Forbidden";
import axiosInstance from "@/axiosInstance";

export default {
  name: "EditWebinar",
  components: {Forbidden},
  setup(){
    const store = useStore();
    const router = useRouter();

    const isForbidden = ref(false)
    const teachers = ref('')
    const methodists = ref('')
    const webinar = ref('')

    const getTeachers = (search, limit, offset) => {
      let queryParams = {
        'search': search,
      };

      if (limit !== null) {
        queryParams.limit = limit;
      }

      if (offset !== null) {
        queryParams.offset = offset;
      }

      axiosInstance.get('/teachers', {params: queryParams})
          .then((response) => {
            console.log(response.data)
            teachers.value = response.data.data;
          })
          .catch((error) => {
            console.log(error.response.data);
          })
    }

    const getMethodists = (search, limit, offset) => {
      let queryParams = {
        'search': search,
      };

      if (limit !== null) {
        queryParams.limit = limit;
      }

      if (offset !== null) {
        queryParams.offset = offset;
      }

      axiosInstance.get('/methodists', {params: queryParams})
          .then((response) => {
            console.log(response.data)
            methodists.value = response.data.data;
          })
          .catch((error) => {
            console.log(error.response.data);
          })
    }

    const getWebinar = (id) => {
      axiosInstance.get(`/webinars/${id}`)
          .then((response) => {
            console.log(response.data)
            webinar.value = response.data;
          })
          .catch((error) => {
            console.log(error.response.data);
          })
    }

    onBeforeMount(() => {
      if (!store.getters.userIsLoggedIn || store.getters.userRole !== 'admin') {
        isForbidden.value = true;
      }
      getWebinar(router.currentRoute.value.params.id)
      getTeachers('', null, null)
      getMethodists('', null, null)
    })

    const searchTeachers = (e) => {
      const form = new FormData(e.target);

      const inputs = Object.fromEntries(form.entries());

      getTeachers(inputs.search, null, null)
    }

    const searchMethodists = (e) => {
      const form = new FormData(e.target);

      const inputs = Object.fromEntries(form.entries());

      getMethodists(inputs.search, null, null)
    }

    const setMethodist = (id) => {
      axiosInstance.put(`/webinars/${webinar.value.id}/methodist/${id}`)
          .then((response) => {
            console.log(response.data)
            getWebinar(webinar.value.id)
          })
          .catch((error) => {
            console.log(error.response.data);
          })
    }

    const setTeacher = (id) => {
      axiosInstance.put(`/webinars/${webinar.value.id}/teacher/${id}`)
          .then((response) => {
            console.log(response.data)
            getWebinar(webinar.value.id)
          })
          .catch((error) => {
            console.log(error.response.data);
          })
    }

    const deleteTeacher = () => {
      axiosInstance.delete(`/webinars/${webinar.value.id}/teacher`)
          .then((response) => {
            console.log(response.data)
            getWebinar(webinar.value.id)
          })
          .catch((error) => {
            console.log(error.response.data);
          })
    }

    const deleteMethodist = () => {
      axiosInstance.delete(`/webinars/${webinar.value.id}/methodist`)
          .then((response) => {
            console.log(response.data)
            getWebinar(webinar.value.id)
          })
          .catch((error) => {
            console.log(error.response.data);
          })
    }

    return{
      teachers,
      methodists,
      webinar,
      isForbidden,
      searchTeachers,
      searchMethodists,
      setTeacher,
      setMethodist,
      deleteTeacher,
      deleteMethodist,
    }
  }
}
</script>

<style scoped>
.card{
  min-width: 350px;
}
</style>