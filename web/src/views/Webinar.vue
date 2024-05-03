<template>
  <div class="m-5">
    <a href="http://localhost:8000/api/excel" class="btn btn-success">Export excel</a>
  </div>
  <main class="mx-auto">
    <div class="container">
      <h1 class="text-center">Количество позитивный и негативных отзывов</h1>
      <canvas ref="positiveNegativeChart"></canvas>
    </div>
    <div class="container">
      <h1 class="text-center">Доля положительных триплетов</h1>
      <canvas ref="positiveTripletsChart"></canvas>
    </div>
    <div class="container">
      <h1 class="text-center">Cамые негативные аспекты</h1>
      <canvas ref="negativeAspectsChart"></canvas>
    </div>
    <div class="container">
      <h1 class="text-center">Cамые позитивные аспекты</h1>
      <canvas ref="positiveAspectsChart"></canvas>
    </div>
    <div class="container">
      <div class="d-flex">
        <select v-model="selectedAspect" class="form-select me-4" aria-label="Default select example">
          <option v-for="aspect in allAspects" :key="aspect.id" :value="aspect">{{aspect}}</option>
        </select>
        <h1 class="text-center">Доля положительных триплетов c аспектом</h1>
      </div>
      <div>
        <canvas ref="positiveTripletsWithAspectChart"></canvas>
      </div>
    </div>
    <div v-for="index in 14" :key="index" class="container image d-flex justify-content-center">
      <img :src="require(`@/assets/images/${index}.png`)">
    </div>
  </main>
</template>

<script>
import Chart from 'chart.js/auto';
import axiosInstance from "@/axiosInstance";
import {ref, watch} from "vue";
import {useRouter} from "vue-router";


export default {
  name: "Webinar",
  setup(){
    const router = useRouter();
    const data = ref('');
    const selectedAspect = ref('');
    const allAspects = ref([]);
    const positiveNegativeChart = ref(null);
    const negativeAspectsChart = ref(null);
    const negativeAspectsChartData = ref({});
    const positiveAspectsChart = ref(null);
    const positiveAspectsChartData = ref({});
    const positiveTripletsChartData = ref({
      labels: [],
      datasets: [
        {
          label: 'Доля положительных триплетов',
          borderColor: '#f87979',
          backgroundColor: 'transparent',
          data: []
        }
      ]
    });
    const positiveTripletsWithAspectChart = ref(null);
    const positiveTripletsChartWithAspectData = ref({
      labels: [],
      datasets: [
        {
          label: 'Доля положительных триплетов c аспектом',
          borderColor: '#f87979',
          backgroundColor: 'transparent',
          data: []
        }
      ]
    });
    const positiveTripletsChart = ref(null);

    const countAspects = (aspects) => {
      return aspects.reduce((acc, aspect) => {
        acc[aspect] = (acc[aspect] || 0) + 1;
        return acc;
      }, {});
    };

    const findPositiveAspects = () => {
      return data.value.aspect.filter((aspect, index) => data.value.sentiment[index] === 'POS');
    };


    const renderPositiveAspectsChart = () => {
      const negativeAspects = findPositiveAspects();
      const aspectCounts = countAspects(negativeAspects);

      const sortedData = Object.entries(aspectCounts)
          .sort((a, b) => b[1] - a[1])
          .slice(0, 5)
          .reduce((obj, [key, value]) => {
            obj[key] = value;
            return obj;
          }, {});

      positiveAspectsChartData.value = {
        labels: Object.keys(sortedData),
        datasets: [
          {
            label: 'Positive Aspects',
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            borderColor: 'rgba(255, 99, 132, 1)',
            borderWidth: 1,
            data: Object.values(sortedData),
          },
        ],
      };
      const ctx = positiveAspectsChart.value.getContext('2d');
      positiveAspectsChart.value = new Chart(ctx, {
        type: 'bar',
        data: positiveAspectsChartData.value,
      });
    }


    const findNegativeAspects = () => {
      return data.value.aspect.filter((aspect, index) => data.value.sentiment[index] === 'NEG');
    };


    const renderNegativeAspectsChart = () => {
      const negativeAspects = findNegativeAspects();
      const aspectCounts = countAspects(negativeAspects);
      negativeAspectsChartData.value = {
        labels: Object.keys(aspectCounts),
        datasets: [
          {
            label: 'Negative Aspects',
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            borderColor: 'rgba(255, 99, 132, 1)',
            borderWidth: 1,
            data: Object.values(aspectCounts),
          },
        ],
      };
      const ctx = negativeAspectsChart.value.getContext('2d');
      negativeAspectsChart.value = new Chart(ctx, {
        type: 'bar',
        data: negativeAspectsChartData.value,
      });
    }

    const calculatePositiveTripletsRatio = (entries) => {
      let totalPositiveTriplets = 0;
      let totalTriplets = 0;
      entries.forEach(entry => {
        totalTriplets += entry.triplets.length;
        totalPositiveTriplets += entry.triplets.filter(triplet => triplet[2] === 'POS').length;
      });
      return totalPositiveTriplets / totalTriplets;
    };

    const renderPositiveTripletsWithAspectChart = () => {
      const groupedData = {};
      // Группируем данные по дате и фильтруем объекты без триплетов
      data.value.data.forEach(entry => {
        if (entry.triplets && entry.triplets.length > 0) {
          if (entry.triplets[0][0] === selectedAspect.value) {
            const date = new Date(entry.created_at).toLocaleDateString();
            if (!groupedData[date]) {
              groupedData[date] = [];
            }
            groupedData[date].push(entry);
          }}
      });
      console.log(groupedData);

      for (const date in groupedData) {
        const positiveTripletsRatio = calculatePositiveTripletsRatio(groupedData[date]);
        positiveTripletsChartWithAspectData.value.labels.push(date);
        console.log(positiveTripletsRatio)
        positiveTripletsChartWithAspectData.value.datasets[0].data.push(positiveTripletsRatio);
      }

      const ctx = positiveTripletsWithAspectChart.value.getContext('2d');
      positiveTripletsWithAspectChart.value = new Chart(ctx, {
        type: 'line',
        data: positiveTripletsChartWithAspectData.value,
      });
    };


    const renderPositiveTripletsChart = () => {
      const groupedData = {};
      // Группируем данные по дате и фильтруем объекты без триплетов
      data.value.data.forEach(entry => {
        if (entry.triplets && entry.triplets.length > 0) {
          const date = new Date(entry.created_at).toLocaleDateString();
          if (!groupedData[date]) {
            groupedData[date] = [];
          }
          groupedData[date].push(entry);
        }
      });
      console.log(groupedData);

      for (const date in groupedData) {
        const positiveTripletsRatio = calculatePositiveTripletsRatio(groupedData[date]);
        positiveTripletsChartData.value.labels.push(date);
        console.log(positiveTripletsRatio)
        positiveTripletsChartData.value.datasets[0].data.push(positiveTripletsRatio);
      }
      const ctx = positiveTripletsChart.value.getContext('2d');
      positiveTripletsChart.value = new Chart(ctx, {
        type: 'line',
        data: positiveTripletsChartData.value,
      });
    };

    const renderPositiveNegativeChart = () => {
      const positiveCount = data.value.data.filter(item => item.sent === 1).length;
      const negativeCount = data.value.data.filter(item => item.sent === 0).length;

      const ctx = positiveNegativeChart.value.getContext('2d');
      positiveNegativeChart.value = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: ['Positive', 'Negative'],
          datasets: [{
            label: 'Sentiment Analysis',
            data: [positiveCount, negativeCount],
            backgroundColor: [
              'rgba(54, 162, 235, 0.2)',
              'rgba(255, 99, 132, 0.2)',
            ],
            borderColor: [
              'rgba(54, 162, 235, 1)',
              'rgba(255, 99, 132, 1)',
            ],
            borderWidth: 1,
          }],
        },
        options: {
          scales: {
            yAxes: [{
              ticks: {
                beginAtZero: true,
              },
            }],
          },
        },
      });
    }


    watch(selectedAspect, async() => {
      console.log(selectedAspect)
      const groupedData = {};
      // Группируем данные по дате и фильтруем объекты без триплетов
      data.value.data.forEach(entry => {
        if (entry.triplets && entry.triplets.length > 0) {
          if (entry.triplets[0][0] === selectedAspect.value) {
            const date = new Date(entry.created_at).toLocaleDateString();
            if (!groupedData[date]) {
              groupedData[date] = [];
            }
            groupedData[date].push(entry);
          }}
      });
      console.log(groupedData);

      positiveTripletsChartWithAspectData.value = {
        labels: [],
        datasets: [
          {
            label: 'Доля положительных триплетов c аспектом',
            borderColor: '#f87979',
            backgroundColor: 'transparent',
            data: []
          }
        ]
      }

      for (const date in groupedData) {
        const positiveTripletsRatio = calculatePositiveTripletsRatio(groupedData[date]);
        positiveTripletsChartWithAspectData.value.labels.push(date);
        console.log(positiveTripletsRatio)
        positiveTripletsChartWithAspectData.value.datasets[0].data.push(positiveTripletsRatio);
      }

      let chartStatus = Chart.getChart(positiveTripletsWithAspectChart.value)
      chartStatus.data = positiveTripletsChartWithAspectData.value
      chartStatus.update()
    })

    axiosInstance.get(`/webinars/${router.currentRoute.value.params.id}/statistics`)
        .then((response) => {
          console.log(response)
          data.value = response.data;
          allAspects.value = [...new Set(data.value.aspect)];
          console.log(allAspects.value)
          renderPositiveNegativeChart()
          renderPositiveTripletsChart()
          renderNegativeAspectsChart()
          renderPositiveAspectsChart()
          renderPositiveTripletsWithAspectChart()
        })
        .catch((error) => {
          console.log(error)
        })

    return{
      positiveNegativeChart,
      positiveTripletsChart,
      negativeAspectsChart,
      positiveAspectsChart,
      selectedAspect,
      allAspects,
      positiveTripletsWithAspectChart,
    }
  }
}
</script>

<style scoped>
.container{
  width: 1000px;
  height: 600px;
}
.form-select {
  height: 50px;
  width: 200px;
}
/*.image{*/
/*  max-height: 400px;*/
/*}*/
.image img{
  width: 1000px;
  height: auto;
  margin-bottom: 50px;
}
</style>