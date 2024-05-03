<template>
  <main v-if="!isForbidden">
    <button @click="showModal = true" class="create-btn btn btn-success" type="submit"> + Create webinar</button>

    <!-- Modal -->
    <div v-if="showModal" class="modal" tabindex="-1" style="display: block">
      <div class="modal-dialog">
        <div class="modal-content">
          <div class="modal-header">
            <h5 class="modal-title">Create webinar</h5>
            <button @click="showModal = false" type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
          </div>
          <form @submit.prevent="createSubmit">
            <div class="modal-body">

              <div class="form-floating">
                <input class="form-control" :class="{'is-invalid': !isValidTitle }" name="title" placeholder="Name">
                <label>Title</label>
              </div>
            </div>
            <div class="modal-footer">
              <button @click="showModal = false" type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
              <button type="submit" class="btn btn-primary">Create</button>
            </div>
          </form>
        </div>
      </div>
    </div>
    <!-- Modal -->

    <div class="items-list mx-auto m-5">
      <form class="search d-flex" @submit.prevent="searchSubmit">
        <input class="form-control mt-2" name="search" placeholder="Type to search...">
        <button class="btn btn-primary ms-2" type="submit"> Search</button>
      </form>
      <div v-for="webinar in webinars" :key="webinar.id" class="card w-30 mt-3">
        <div class="card-body">
          <h5 class="card-title">ID: {{webinar.id}}</h5>
          <p class="card-text">Title: {{webinar.title}}</p>
          <p v-if="webinar.methodist_id !== null" class="card-text">Methodist: {{webinar.methodist.firstname}} {{webinar.methodist.lastname}}</p>
          <p v-else class="card-text">Methodist: not specified</p>
          <p v-if="webinar.teacher_id !== null" class="card-text">Teacher: {{webinar.teacher.firstname}} {{webinar.teacher.lastname}}</p>
          <p v-else class="card-text">Teacher: not specified</p>
          <div class="d-flex">
            <button @click="router.push(`/webinar/${webinar.id}`)" class="btn btn-primary me-2">Get statistics</button>
            <button @click="router.push(`/edit-webinar/${webinar.id}`)" type="button" class="btn btn-success me-2">Edit</button>
            <button @click="deleteSubmit(webinar.id)" type="button" class="btn btn-danger me-2">Delete</button>
          </div>
        </div>
      </div>
    </div>
  </main>
  <Forbidden v-else />
</template>

<script>
import {useStore} from "vuex";
import axiosInstance from "@/axiosInstance";
import {ref, onBeforeMount} from "vue";
import Forbidden from "@/views/Forbidden";
import {useRouter} from "vue-router";


export default {
  name: "AdminPanel",
  components: {Forbidden},
  setup() {
    const store = useStore();
    const router = useRouter();

    const webinars = ref([]);
    const showModal = ref(false);
    const isValidTitle = ref(true);
    const isForbidden = ref(false);

    const userIsLoggedIn = store.getters.userIsLoggedIn
    const userRole = store.getters.userRole

    const getWebinars = (search, limit, offset) => {
      let queryParams = {
        'search': search,
      };

      if (limit !== null) {
        queryParams.limit = limit;
      }

      if (offset !== null) {
        queryParams.offset = offset;
      }

      axiosInstance.get('/webinars', {params: queryParams})
          .then((response) => {
            console.log(response.data)
            webinars.value = response.data.data;
          })
          .catch((error) => {
            console.log(error.response.data);
          })
    }

    const searchSubmit = (e) => {
      const form = new FormData(e.target);

      const inputs = Object.fromEntries(form.entries());

      getWebinars(inputs.search, null, null)
    }

    const createSubmit = (e) => {
      const form = new FormData(e.target);

      const inputs = Object.fromEntries(form.entries());

      axiosInstance.post('/webinars', inputs)
          .then((response) => {
            webinars.value.push(response.data);
            showModal.value = false;
          })
          .catch((error) => {
            console.log(error.response.data)
          })

    }

    const deleteSubmit = (item_id) => {
      axiosInstance.delete(`/webinars/${item_id}`)
          .then(() => {
            getWebinars('', null, null)
          })
          .catch((error) => {
            console.log(error.response.data)
          })
    }

    onBeforeMount(() => {
      getWebinars('', null, null)
    })

    return {
      userIsLoggedIn,
      userRole,
      searchSubmit,
      webinars,
      showModal,
      isValidTitle,
      createSubmit,
      deleteSubmit,
      isForbidden,
      router,
    }
  }
}


</script>

<style scoped>

.items-list {
  width: 100%;
  max-width: 500px;
}

.create-btn {
  margin-top: 50px;
  margin-left: 1vw;
}

.form-floating, .btn {
  margin-top: 10px;
}

</style>