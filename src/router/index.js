import { createRouter, createWebHashHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    name: 'index',
    component: () => import("../views/MainView.vue")
  },
  {
    path: '/edit',
    name: 'edit view',
    component: () => import("../views/MainView.vue")
  },
  {
    path: '/display',
    name: 'edit view',
    component: () => import("../views/DisplayView.vue")
  },
]

const router = createRouter({
  history: createWebHashHistory(),
  routes
})

export default router